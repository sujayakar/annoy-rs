use std::{
    collections::HashMap,
    env, fs,
    path::{Path, PathBuf},
    sync::atomic::AtomicBool,
    time::Instant,
};

use qdrant_segment::{
    data_types::named_vectors::NamedVectors,
    entry::entry_point::SegmentEntry,
    segment_constructor::{build_segment, load_segment, segment_builder::SegmentBuilder},
    types::{
        Condition, Distance, FieldCondition, Filter, HnswConfig, Indexes, Match, MatchValue,
        PayloadStorageType, PointIdType, SearchParams, SegmentConfig, ValueVariants,
        VectorDataConfig, VectorStorageType, WithPayload, WithVector, DEFAULT_FULL_SCAN_THRESHOLD,
        DEFAULT_HNSW_EF_CONSTRUCT,
    },
};
use rand::Rng;
use serde_json::json;
use tempdir::TempDir;
use uuid::Uuid;

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

fn create_disk_index(
    out_dir: &Path,
    num_vectors: usize,
    num_users: usize,
) -> anyhow::Result<PathBuf> {
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
    for _ in 0..num_vectors {
        let seq_num = 1;
        let point_id = PointIdType::Uuid(random_uuid(&mut rng));
        let vector = random_normalized_vector(&mut rng);
        let named_vectors = NamedVectors::from_ref(VECTOR_NAME, &vector);
        memory_segment.upsert_point(seq_num, point_id, named_vectors)?;

        let payload = json!({"userId": rng.gen_range(0..num_users)}).into();
        memory_segment.set_payload(seq_num, point_id, &payload)?;
    }
    println!(
        "Inserted {} random vectors into memory index in {:?}",
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

fn query(path: &Path, num_results: usize, user_id: Option<usize>) -> anyhow::Result<()> {
    let start = Instant::now();
    let Some(segment) = load_segment(path)? else {
        anyhow::bail!("Failed to load segment: Segment not properly saved");
    };
    println!("Loaded segment in {:?}", start.elapsed());
    let query_vector = random_normalized_vector(&mut rand::thread_rng());
    let with_payload = WithPayload {
        enable: false,
        payload_selector: None,
    };
    let with_vector = WithVector::Bool(false);
    let mut filter = Filter {
        should: None,
        must: None,
        must_not: None,
    };
    if let Some(user_id) = user_id {
        let condition = Condition::Field(FieldCondition::new_match(
            "userId",
            Match::Value(MatchValue {
                value: ValueVariants::Integer(user_id as i64),
            }),
        ));
        filter.should = Some(vec![condition]);
        // TODO: Find a way to get at the query planner when we add a filter.
    }
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
        .ok_or_else(|| anyhow::anyhow!("Usage: ./qdrant-test (create|query|merge)"))?;
    match &command[..] {
        "create" => {
            let path = env::args().nth(2).ok_or_else(|| {
                anyhow::anyhow!("Usage: ./qdrant-test create <path> <numVectors> <numUsers>")
            })?;
            let num_vectors = env::args()
                .nth(3)
                .map(|n| n.parse::<usize>())
                .transpose()?
                .unwrap_or(1000);
            let num_users = env::args()
                .nth(4)
                .map(|n| n.parse::<usize>())
                .transpose()?
                .unwrap_or(100);
            create_disk_index(&Path::new(&path), num_vectors, num_users)?;
        }
        "query" => {
            let path = env::args().nth(2).ok_or_else(|| {
                anyhow::anyhow!("Usage: ./qdrant-test query <path> <numResults> <userId>")
            })?;
            let num_results = env::args()
                .nth(3)
                .map(|n| n.parse::<usize>())
                .transpose()?
                .unwrap_or(5);
            let user_id = env::args().nth(4).map(|n| n.parse::<usize>()).transpose()?;
            query(&Path::new(&path), num_results, user_id)?;
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
