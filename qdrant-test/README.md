First, build the binary.

```
$ RUSTFLAGS="-C target-cpu=native" cargo build --release
```

Then, create a new disk index in `data/out1` with 10000 vectors random vectors of dimension 1536 and 100 distinct user IDs for filtering.

```
$ ./target/release/qdrant-test create data/out1 10000 100
Generated 10000 vectors in 117.264625ms
Inserted 10000 random vectors into memory index in 177.687625ms
Built disk index (10000 points) in 4.687438625s
```

For some reason, it outputs the disk index underneath the output directory, so you'll have to find the path.

```
$ ls data/out1/disk
2744ef04-0e12-47bb-9385-a9a700352735
```

Query this index for the top 10 results closest to a random vector.

```
$ ./target/release/qdrant-test query data/out1/disk/2744ef04-0e12-47bb-9385-a9a700352735 10
Loaded segment in 15.423542ms
Queried 10 of 10 results in 2.868792ms:
  Uuid(2ec6d869-c7f6-c08e-58ab-db1628cc9bf3): 0.772764
  Uuid(ca87a289-b0b9-f2b1-e522-72d8d7eb537a): 0.7720382
  Uuid(211b7b2d-3275-4b90-eb46-0346b6bc54a8): 0.7715249
  Uuid(aba73fd1-f70e-e6db-5b6e-9b9c62c9bce9): 0.7713932
  Uuid(be18c2d7-c137-cbcf-195e-f2b73c3b1204): 0.7710824
  Uuid(0d307050-0c91-3161-e84d-965bc40174a2): 0.77097607
  Uuid(d5e0fbaa-ede8-540b-bd38-5a33af086f55): 0.77057636
  Uuid(ee8b3f2a-8689-d8f3-5925-8a399c29045b): 0.77021754
  Uuid(f05878ea-66d6-a2cf-49ee-9c691fbed70d): 0.7701825
  Uuid(20ca8ea5-43fc-b9ab-fa8d-936e7d89ed92): 0.77013814
```

Perform the same query filtering for a specific user ID.

```
$ ./target/release/qdrant-test query data/out1/disk/2744ef04-0e12-47bb-9385-a9a700352735 10 24
Loaded segment in 21.42575ms
Queried 10 of 10 results in 15.760583ms:
  Uuid(aeaff0ed-2309-5cfa-a8ff-60a6265c431b): 0.7688428
  Uuid(6f4a9b65-dac6-f4c3-94dd-027c174d57f2): 0.7688328
  Uuid(fd54d342-e687-88c6-a20c-35d43fcfc172): 0.7672464
  Uuid(661d1df0-cfdf-0d74-94db-4e6d0513abca): 0.76252645
  Uuid(0e81f1ca-ac83-0ce1-ce73-9131cd5324f4): 0.7611766
  Uuid(acb1bbdb-264f-18a3-f10d-b88a36f3300b): 0.7611457
  Uuid(5474db30-0aba-4b10-b5ba-263a2537c289): 0.76073
  Uuid(da933b85-300e-9aba-b989-6ec194c97835): 0.76053077
  Uuid(08c1296e-68d6-3001-7537-b9a73b4b988b): 0.7601761
  Uuid(999a3419-c4fd-8d28-243c-5ef056fd7526): 0.75972015
```

Build another index and merge it into a third index.

```
$ ./target/release/qdrant-test create data/out2 10000 100
$ ./target/release/qdrant-test merge data/out1/disk/2744ef04-0e12-47bb-9385-a9a700352735 data/out2/disk/cc1895c1-4ac3-47ff-ad69-0484cff76c4c data/out3
Merged disk indexes (total 20000 points) in 11.177887834s
```
