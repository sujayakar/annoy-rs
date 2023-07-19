use std::{
    collections::HashMap,
    env, fs,
    path::{Path, PathBuf},
    sync::{atomic::AtomicBool, Arc},
    time::Instant,
};

use atomic_refcell::AtomicRefCell;
use parking_lot::Mutex;
use qdrant_segment::{
    common::{
        rocksdb_wrapper::{open_db, DB_VECTOR_CF},
        version::StorageVersion,
    },
    data_types::named_vectors::NamedVectors,
    entry::entry_point::SegmentEntry,
    id_tracker::{simple_id_tracker::SimpleIdTracker, IdTracker},
    index::{
        hnsw_index::{
            graph_links::{GraphLinksMmap, GraphLinksRam},
            hnsw::HNSWIndex,
        },
        plain_payload_index::PlainIndex,
        struct_payload_index::StructPayloadIndex,
        VectorIndexEnum,
    },
    payload_storage::{
        on_disk_payload_storage::OnDiskPayloadStorage, simple_payload_storage::SimplePayloadStorage,
    },
    segment::{Segment, SegmentVersion, VectorData},
    segment_constructor::{
        get_vector_index_path, get_vector_storage_path, load_segment,
        segment_builder::SegmentBuilder, PAYLOAD_INDEX_PATH, build_segment,
    },
    types::{
        Distance, Filter, HnswConfig, Indexes, PayloadStorageType, PointIdType, SearchParams,
        SegmentConfig, SegmentType, VectorDataConfig, VectorStorageType, WithPayload, WithVector,
        DEFAULT_FULL_SCAN_THRESHOLD, DEFAULT_HNSW_EF_CONSTRUCT,
    },
    vector_storage::{
        appendable_mmap_vector_storage::open_appendable_memmap_vector_storage,
        memmap_vector_storage::open_memmap_vector_storage,
        simple_vector_storage::open_simple_vector_storage, VectorStorage,
    },
};
use rand::Rng;
use tempdir::TempDir;
use uuid::Uuid;

fn sp<T>(t: T) -> Arc<AtomicRefCell<T>> {
    Arc::new(AtomicRefCell::new(t))
}

const VECTOR_NAME: &str = "vector_name";
const DIMENSION: usize = 1536;

fn segment_config(append: bool) -> SegmentConfig {
    let index = if append {
        Indexes::Plain {}
    } else {
        let hnsw_config = HnswConfig {
            /// Number of edges per node in the index graph. Larger the value -
            /// more accurate the search, more space required.
            m: 16,
            /// Number of neighbours to consider during the index building. Larger
            ///  the value - more accurate the search, more time required to build index.
            ef_construct: DEFAULT_HNSW_EF_CONSTRUCT,
            /// Minimal size (in KiloBytes) of vectors for additional payload-based indexing.
            /// If payload chunk is smaller than `full_scan_threshold_kb` additional indexing
            /// won't be used - in this case full-scan search should be preferred by query
            /// planner and additional indexing is not required.
            /// Note: 1Kb = 1 vector of size 256
            full_scan_threshold: DEFAULT_FULL_SCAN_THRESHOLD,
            /// Number of parallel threads used for background index building. If 0 - auto
            /// selection.
            max_indexing_threads: 4,
            on_disk: Some(true),
            /// Custom M param for hnsw graph built for payload index. If not set, default M
            /// will be used.
            payload_m: None,
        };
        Indexes::Hnsw(hnsw_config)
    };
    let vector_storage_type = if append {
        VectorStorageType::Memory
    } else {
        VectorStorageType::Mmap
    };
    let payload_storage_type = if append {
        PayloadStorageType::InMemory
    } else {
        PayloadStorageType::OnDisk
    };
    let vector_data_config = VectorDataConfig {
        size: DIMENSION,
        distance: Distance::Cosine,
        storage_type: vector_storage_type,
        index,
        quantization_config: None,
    };
    SegmentConfig {
        vector_data: HashMap::from([(VECTOR_NAME.to_string(), vector_data_config)]),
        payload_storage_type,
    }
}

fn get_vector_name_with_prefix(prefix: &str, vector_name: &str) -> String {
    if !vector_name.is_empty() {
        format!("{prefix}-{vector_name}")
    } else {
        prefix.to_owned()
    }
}

fn random_uuid(rng: &mut impl Rng) -> Uuid {
    Uuid::from_bytes(rng.gen())
}

fn random_normalized_vector(rng: &mut impl Rng) -> Vec<f32> {
    let mut v = vec![0f32; 1536];
    for i in 0..1536 {
        v[i] = rng.gen();
    }
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    for i in 0..1536 {
        v[i] /= norm;
    }
    v
}

fn create_disk_index(out_dir: &Path) -> anyhow::Result<PathBuf> {
    let mut rng = rand::thread_rng();
    let start = Instant::now();
    let data: Vec<_> = (0..10_000)
        .map(|_| (random_uuid(&mut rng), random_normalized_vector(&mut rng)))
        .collect();
    println!("Generated {} vectors in {:?}", data.len(), start.elapsed());

    fs::create_dir_all(&out_dir)?;

    let scratch_dir = out_dir.join("_scratch");
    fs::create_dir_all(&scratch_dir)?;

    let memory_config = segment_config(true);
    let memory_segment_path = out_dir.join("memory");
    let mut memory_segment = build_segment(&memory_segment_path, &memory_config, true)?;
    // memory_segment.save_current_state()?;
    // SegmentVersion::save(&memory_segment_path)?;
    // fill_segment1(&mut memory_segment)?;

    let start = Instant::now();
    for (uuid, v) in &data {
        memory_segment.upsert_point(
            1,
            PointIdType::Uuid(*uuid),
            NamedVectors::from_ref(VECTOR_NAME, v),
        )?;
    }
    println!(
        "Inserted {} vectors into memory index in {:?}",
        data.len(),
        start.elapsed()
    );

    let start = Instant::now();

    // Pack the memory segment into a disk segment.
    let stopped = AtomicBool::new(false);
    let disk_segment_path = out_dir.join("disk");
    fs::create_dir_all(&disk_segment_path)?;
    let disk_config = segment_config(false);
    let mut builder = SegmentBuilder::new(&disk_segment_path, &scratch_dir, &disk_config)?;
    builder.update_from(&memory_segment, &stopped)?;
    let disk_segment = builder.build(&stopped)?;
    disk_segment.save_current_state()?;

    println!(
        "Built disk index ({} points) in {:?}",
        disk_segment.iter_points().count(),
        start.elapsed()
    );

    Ok(disk_segment_path)
}

fn query(path: &Path, num_results: usize) -> anyhow::Result<()> {
    let start = Instant::now();
    let Some(segment) = load_segment(path)? else {
        anyhow::bail!("Failed to load segment: Segment not properly saved");
    };
    println!(
        "Loaded segment in {:?} with config: {:#?}",
        start.elapsed(),
        segment.segment_config
    );
    let query_vector = random_normalized_vector(&mut rand::thread_rng());
    let with_payload = WithPayload {
        enable: false,
        payload_selector: None,
    };
    let with_vector = WithVector::Bool(false);
    let filter = Filter {
        should: None,
        must: None,
        must_not: None,
    };
    let search_params = SearchParams {
        hnsw_ef: None,
        exact: false,
        quantization: None,
    };
    let start = Instant::now();
    let results = segment.search(
        VECTOR_NAME,
        &query_vector,
        &with_payload,
        &with_vector,
        Some(&filter),
        num_results,
        Some(&search_params),
    )?;
    println!(
        "Queried {} of {num_results} results in {:?}:",
        results.len(),
        start.elapsed()
    );
    for result in results {
        println!("  {:?}: {}", result.id, result.score);
    }
    Ok(())
}

fn merge(left_path: &Path, right_path: &Path, out_path: &Path) -> anyhow::Result<()> {
    let Some(left) = load_segment(&left_path)? else {
        anyhow::bail!("Failed to load {left_path:?}");
    };
    let Some(right) = load_segment(&right_path)? else {
        anyhow::bail!("Failed to load {right_path:?}");
    };

    let tmpdir = TempDir::new("qdrant-merge")?;

    let start = Instant::now();
    fs::create_dir_all(&out_path)?;
    let stopped = AtomicBool::new(false);
    let disk_config = segment_config(false);
    let mut builder = SegmentBuilder::new(&out_path, &tmpdir.path(), &disk_config)?;
    builder.update_from(&left, &stopped)?;
    builder.update_from(&right, &stopped)?;
    let merged_segment = builder.build(&stopped)?;
    merged_segment.save_current_state()?;
    println!(
        "Merged disk indexes (total {} points) in {:?}",
        merged_segment.iter_points().count(),
        start.elapsed()
    );
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let command = env::args()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("Usage: ./qdrant-test (create <path> | query <path>)"))?;
    match &command[..] {
        "create" => {
            let path = env::args()
                .nth(2)
                .ok_or_else(|| anyhow::anyhow!("Usage: ./qdrant-test create <path>"))?;
            create_disk_index(&Path::new(&path))?;
        }
        "query" => {
            let path = env::args()
                .nth(2)
                .ok_or_else(|| anyhow::anyhow!("Usage: ./qdrant-test query <path> <numResults>"))?;
            let num_results: usize = env::args()
                .nth(3)
                .map(|n| n.parse::<usize>())
                .transpose()?
                .unwrap_or(5);
            query(&Path::new(&path), num_results)?;
        }
        // See collection/collection_manager/optimizers for more details on merge policies.
        "merge" => {
            let out_path = env::args()
                .nth(4)
                .ok_or_else(|| anyhow::anyhow!("Usage: ./qdrant-test merge <in> <in> <out>"))?;
            let right_path = env::args().nth(3).unwrap();
            let left_path = env::args().nth(2).unwrap();
            merge(
                &Path::new(&left_path),
                &Path::new(&right_path),
                &Path::new(&out_path),
            )?;
        }
        s => anyhow::bail!("Unsupported command: {s}"),
    }
    Ok(())
}

// fn fill_segment1_dim4(segment: &mut Segment) -> anyhow::Result<()> {
//     let vec1 = vec![1.0, 0.0, 1.0, 1.0];
//     let vec2 = vec![1.0, 0.0, 1.0, 0.0];
//     let vec3 = vec![1.0, 1.0, 1.0, 1.0];
//     let vec4 = vec![1.0, 1.0, 0.0, 1.0];
//     let vec5 = vec![1.0, 0.0, 0.0, 0.0];

//     segment.upsert_point(1, 1.into(), NamedVectors::from_ref(VECTOR_NAME, &vec1))?;
//     segment.upsert_point(2, 2.into(), NamedVectors::from_ref(VECTOR_NAME, &vec2))?;
//     segment.upsert_point(3, 3.into(), NamedVectors::from_ref(VECTOR_NAME, &vec3))?;
//     segment.upsert_point(4, 4.into(), NamedVectors::from_ref(VECTOR_NAME, &vec4))?;
//     segment.upsert_point(5, 5.into(), NamedVectors::from_ref(VECTOR_NAME, &vec5))?;

//     let payload_key = "color";
//     let payload_option1 = json!({ payload_key: vec!["red".to_owned()] }).into();
//     let payload_option2 = json!({ payload_key: vec!["red".to_owned(), "blue".to_owned()] }).into();
//     let payload_option3 = json!({ payload_key: vec!["blue".to_owned()] }).into();

//     segment.set_payload(6, 1.into(), &payload_option1)?;
//     segment.set_payload(6, 2.into(), &payload_option1)?;
//     segment.set_payload(6, 3.into(), &payload_option3)?;
//     segment.set_payload(6, 4.into(), &payload_option2)?;
//     segment.set_payload(6, 5.into(), &payload_option2)?;

//     Ok(())
// }

// fn empty_segment(segment_path: &Path, config: SegmentConfig) -> anyhow::Result<Segment> {
//     fs::create_dir_all(segment_path)?;
//     let vector_db_names: Vec<String> = config
//         .vector_data
//         .keys()
//         .map(|name| format!("vector-{name}"))
//         .collect();
//     let database = open_db(&segment_path, &vector_db_names)?;
//     let payload_storage = match config.payload_storage_type {
//         PayloadStorageType::InMemory => sp(SimplePayloadStorage::open(database.clone())?.into()),
//         PayloadStorageType::OnDisk => sp(OnDiskPayloadStorage::open(database.clone())?.into()),
//     };

//     let id_tracker = sp(SimpleIdTracker::open(database.clone())?);

//     let payload_index_path = segment_path.join(PAYLOAD_INDEX_PATH);
//     let payload_index: Arc<AtomicRefCell<StructPayloadIndex>> = sp(StructPayloadIndex::open(
//         payload_storage,
//         id_tracker.clone(),
//         &payload_index_path,
//     )?);

//     let mut vector_data = HashMap::new();
//     for (vector_name, vector_config) in &config.vector_data {
//         let vector_storage_path = get_vector_storage_path(&segment_path, vector_name);
//         let vector_index_path = get_vector_index_path(&segment_path, vector_name);
//         let vector_storage = match vector_config.storage_type {
//             // In memory
//             VectorStorageType::Memory => {
//                 let db_column_name = get_vector_name_with_prefix(DB_VECTOR_CF, vector_name);
//                 open_simple_vector_storage(
//                     database.clone(),
//                     &db_column_name,
//                     vector_config.size,
//                     vector_config.distance,
//                 )?
//             }
//             // Mmap on disk, not appendable
//             VectorStorageType::Mmap => open_memmap_vector_storage(
//                 &vector_storage_path,
//                 vector_config.size,
//                 vector_config.distance,
//             )?,
//             // Chunked mmap on disk, appendable
//             VectorStorageType::ChunkedMmap => open_appendable_memmap_vector_storage(
//                 &vector_storage_path,
//                 vector_config.size,
//                 vector_config.distance,
//             )?,
//         };
//         let point_count = id_tracker.borrow().total_point_count();
//         let vector_count = vector_storage.borrow().total_vector_count();
//         anyhow::ensure!(point_count == vector_count);
//         anyhow::ensure!(config.quantization_config(vector_name).is_none());
//         let vector_index: Arc<AtomicRefCell<VectorIndexEnum>> = match &vector_config.index {
//             Indexes::Plain {} => sp(VectorIndexEnum::Plain(PlainIndex::new(
//                 id_tracker.clone(),
//                 vector_storage.clone(),
//                 payload_index.clone(),
//             ))),
//             Indexes::Hnsw(vector_hnsw_config) => sp(if vector_hnsw_config.on_disk == Some(true) {
//                 VectorIndexEnum::HnswMmap(HNSWIndex::<GraphLinksMmap>::open(
//                     &vector_index_path,
//                     id_tracker.clone(),
//                     vector_storage.clone(),
//                     payload_index.clone(),
//                     vector_hnsw_config.clone(),
//                 )?)
//             } else {
//                 VectorIndexEnum::HnswRam(HNSWIndex::<GraphLinksRam>::open(
//                     &vector_index_path,
//                     id_tracker.clone(),
//                     vector_storage.clone(),
//                     payload_index.clone(),
//                     vector_hnsw_config.clone(),
//                 )?)
//             }),
//         };
//         vector_data.insert(
//             vector_name.to_owned(),
//             VectorData {
//                 vector_storage,
//                 vector_index,
//             },
//         );
//     }
//     let segment_type = if config.is_any_vector_indexed() {
//         SegmentType::Indexed
//     } else {
//         SegmentType::Plain
//     };
//     let appendable_flag = vector_data.values().all(VectorData::is_appendable);

//     Ok(Segment {
//         version: None,
//         persisted_version: Arc::new(Mutex::new(None)),
//         current_path: segment_path.to_owned(),
//         id_tracker,
//         vector_data,
//         segment_type,
//         appendable_flag,
//         payload_index,
//         segment_config: config.clone(),
//         error_status: None,
//         database,
//         flush_thread: Mutex::new(None),
//     })
// }
