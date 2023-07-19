fn fill_segment1_dim4(segment: &mut Segment) -> anyhow::Result<()> {
    let vec1 = vec![1.0, 0.0, 1.0, 1.0];
    let vec2 = vec![1.0, 0.0, 1.0, 0.0];
    let vec3 = vec![1.0, 1.0, 1.0, 1.0];
    let vec4 = vec![1.0, 1.0, 0.0, 1.0];
    let vec5 = vec![1.0, 0.0, 0.0, 0.0];

    segment.upsert_point(1, 1.into(), NamedVectors::from_ref(VECTOR_NAME, &vec1))?;
    segment.upsert_point(2, 2.into(), NamedVectors::from_ref(VECTOR_NAME, &vec2))?;
    segment.upsert_point(3, 3.into(), NamedVectors::from_ref(VECTOR_NAME, &vec3))?;
    segment.upsert_point(4, 4.into(), NamedVectors::from_ref(VECTOR_NAME, &vec4))?;
    segment.upsert_point(5, 5.into(), NamedVectors::from_ref(VECTOR_NAME, &vec5))?;

    let payload_key = "color";
    let payload_option1 = json!({ payload_key: vec!["red".to_owned()] }).into();
    let payload_option2 = json!({ payload_key: vec!["red".to_owned(), "blue".to_owned()] }).into();
    let payload_option3 = json!({ payload_key: vec!["blue".to_owned()] }).into();

    segment.set_payload(6, 1.into(), &payload_option1)?;
    segment.set_payload(6, 2.into(), &payload_option1)?;
    segment.set_payload(6, 3.into(), &payload_option3)?;
    segment.set_payload(6, 4.into(), &payload_option2)?;
    segment.set_payload(6, 5.into(), &payload_option2)?;

    Ok(())
}

fn empty_segment(segment_path: &Path, config: SegmentConfig) -> anyhow::Result<Segment> {
    fs::create_dir_all(segment_path)?;
    let vector_db_names: Vec<String> = config
        .vector_data
        .keys()
        .map(|name| format!("vector-{name}"))
        .collect();
    let database = open_db(&segment_path, &vector_db_names)?;
    let payload_storage = match config.payload_storage_type {
        PayloadStorageType::InMemory => sp(SimplePayloadStorage::open(database.clone())?.into()),
        PayloadStorageType::OnDisk => sp(OnDiskPayloadStorage::open(database.clone())?.into()),
    };

    let id_tracker = sp(SimpleIdTracker::open(database.clone())?);

    let payload_index_path = segment_path.join(PAYLOAD_INDEX_PATH);
    let payload_index: Arc<AtomicRefCell<StructPayloadIndex>> = sp(StructPayloadIndex::open(
        payload_storage,
        id_tracker.clone(),
        &payload_index_path,
    )?);

    let mut vector_data = HashMap::new();
    for (vector_name, vector_config) in &config.vector_data {
        let vector_storage_path = get_vector_storage_path(&segment_path, vector_name);
        let vector_index_path = get_vector_index_path(&segment_path, vector_name);
        let vector_storage = match vector_config.storage_type {
            // In memory
            VectorStorageType::Memory => {
                let db_column_name = get_vector_name_with_prefix(DB_VECTOR_CF, vector_name);
                open_simple_vector_storage(
                    database.clone(),
                    &db_column_name,
                    vector_config.size,
                    vector_config.distance,
                )?
            }
            // Mmap on disk, not appendable
            VectorStorageType::Mmap => open_memmap_vector_storage(
                &vector_storage_path,
                vector_config.size,
                vector_config.distance,
            )?,
            // Chunked mmap on disk, appendable
            VectorStorageType::ChunkedMmap => open_appendable_memmap_vector_storage(
                &vector_storage_path,
                vector_config.size,
                vector_config.distance,
            )?,
        };
        let point_count = id_tracker.borrow().total_point_count();
        let vector_count = vector_storage.borrow().total_vector_count();
        anyhow::ensure!(point_count == vector_count);
        anyhow::ensure!(config.quantization_config(vector_name).is_none());
        let vector_index: Arc<AtomicRefCell<VectorIndexEnum>> = match &vector_config.index {
            Indexes::Plain {} => sp(VectorIndexEnum::Plain(PlainIndex::new(
                id_tracker.clone(),
                vector_storage.clone(),
                payload_index.clone(),
            ))),
            Indexes::Hnsw(vector_hnsw_config) => sp(if vector_hnsw_config.on_disk == Some(true) {
                VectorIndexEnum::HnswMmap(HNSWIndex::<GraphLinksMmap>::open(
                    &vector_index_path,
                    id_tracker.clone(),
                    vector_storage.clone(),
                    payload_index.clone(),
                    vector_hnsw_config.clone(),
                )?)
            } else {
                VectorIndexEnum::HnswRam(HNSWIndex::<GraphLinksRam>::open(
                    &vector_index_path,
                    id_tracker.clone(),
                    vector_storage.clone(),
                    payload_index.clone(),
                    vector_hnsw_config.clone(),
                )?)
            }),
        };
        vector_data.insert(
            vector_name.to_owned(),
            VectorData {
                vector_storage,
                vector_index,
            },
        );
    }
    let segment_type = if config.is_any_vector_indexed() {
        SegmentType::Indexed
    } else {
        SegmentType::Plain
    };
    let appendable_flag = vector_data.values().all(VectorData::is_appendable);

    Ok(Segment {
        version: None,
        persisted_version: Arc::new(Mutex::new(None)),
        current_path: segment_path.to_owned(),
        id_tracker,
        vector_data,
        segment_type,
        appendable_flag,
        payload_index,
        segment_config: config.clone(),
        error_status: None,
        database,
        flush_thread: Mutex::new(None),
    })
}
