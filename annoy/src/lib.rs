use std::{
    ffi::{c_char, c_int, c_void, CStr, CString},
    path::Path,
    ptr,
};

// TODO:
// - set seed for determinism
// - check determinism
// - port accuracy test
// - builder pattern
// - change header to use const ptrs where appropraite
// - get_n_trees
// - more rusty APIs than -1 isize
//
// glove-100-angular:
// num_trees: 100-400, search_k: 100,000

use annoy_sys::*;

pub struct AnnoyAngular {
    ptr: *mut c_void,
    dimension: usize,
}

impl Drop for AnnoyAngular {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                annoy_angular_free_index(self.ptr);
                self.ptr = ptr::null_mut();
            }
        }
    }
}

impl AnnoyAngular {
    // AnnoyIndex(f, metric) returns a new index that's read-write and stores vector
    // of f dimensions. Metric can be "angular", "euclidean", "manhattan", "hamming",
    // or "dot".
    pub fn new(dimension: usize) -> Self {
        let ptr = unsafe { annoy_angular_create_index(dimension as c_int) };
        Self { ptr, dimension }
    }

    // a.add_item(i, v) adds item i (any nonnegative integer) with vector v. Note that
    // it will allocate memory for max(i)+1 items.
    pub fn add_item(&mut self, item: u32, vector: &[f32]) -> anyhow::Result<()> {
        anyhow::ensure!(vector.len() == self.dimension);
        assert_eq!(vector.len(), self.dimension);
        unsafe {
            let mut error_ptr: *mut c_char = ptr::null_mut();
            let success = annoy_angular_add_item(
                self.ptr,
                item as c_int,
                vector.as_ptr() as *mut _,
                &mut error_ptr as *mut _,
            );
            check_error("add_item", success, error_ptr)?;
        }
        Ok(())
    }

    // a.build(n_trees, n_jobs=-1) builds a forest of n_trees trees. More trees gives higher
    // precision when querying. After calling build, no more items can be added. n_jobs
    // specifies the number of threads used to build the trees. n_jobs=-1 uses all available
    // CPU cores.
    pub fn build(&mut self, n_trees: i32) -> anyhow::Result<()> {
        unsafe {
            let mut error_ptr: *mut c_char = ptr::null_mut();
            let success =
                annoy_angular_build(self.ptr, n_trees as c_int, 1, &mut error_ptr as *mut _);
            check_error("build", success, error_ptr)?;
        }
        Ok(())
    }

    // a.save(fn, prefault=False) saves the index to disk and loads it (see next function). After
    // saving, no more items can be added.
    pub fn save(&mut self, p: &Path) -> anyhow::Result<()> {
        let p_str = p
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("Path {p:?} is not valid UTF-8"))?;
        let p_cstr = CString::new(p_str)?;
        unsafe {
            let mut error_ptr: *mut c_char = ptr::null_mut();
            let success = annoy_angular_save(
                self.ptr,
                p_cstr.as_ptr() as *mut _,
                false,
                &mut error_ptr as *mut _,
            );
            check_error("save", success, error_ptr)?;
        }
        Ok(())
    }

    // a.load(fn, prefault=False) loads (mmaps) an index from disk. If prefault is set to True, it
    // will pre-read the entire file into memory (using mmap with MAP_POPULATE). Default is False.
    pub fn load(&mut self, p: &Path) -> anyhow::Result<()> {
        let p_str = p
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("Path {p:?} is not valid UTF-8"))?;
        let p_cstr = CString::new(p_str)?;
        unsafe {
            let mut error_ptr: *mut c_char = ptr::null_mut();
            let success = annoy_angular_load(
                self.ptr,
                p_cstr.as_ptr() as *mut _,
                false,
                &mut error_ptr as *mut _,
            );
            check_error("load", success, error_ptr)?;
        }
        Ok(())
    }

    // a.unload() unloads.
    pub fn unload(&mut self) {
        unsafe {
            annoy_angular_unload(self.ptr);
        }
    }

    // a.get_nns_by_item(i, n, search_k=-1, include_distances=False) returns the n closest items.
    // During the query it will inspect up to search_k nodes which defaults to n_trees * n if not
    // provided. search_k gives you a run-time tradeoff between better accuracy and speed. If you
    // set include_distances to True, it will return a 2 element tuple with two lists in it: the
    // second one containing all corresponding distances.
    pub fn get_nearest_by_item(
        &mut self,
        item: u32,
        n: usize,
        search_k: i32,
    ) -> anyhow::Result<(Vec<u32>, Vec<f32>)> {
        // TODO: bounds checking?
        unsafe {
            let mut results = Vec::with_capacity(n);
            let mut distances = Vec::with_capacity(n);
            let num_results = annoy_angular_get_nns_by_item(
                self.ptr,
                item,
                n,
                search_k,
                results.as_mut_ptr(),
                distances.as_mut_ptr(),
            );
            results.set_len(num_results);
            distances.set_len(num_results);
            Ok((results, distances))
        }
    }

    // a.get_nns_by_vector(v, n, search_k=-1, include_distances=False) same but query by vector v.
    pub fn get_nearest_by_vector(
        &mut self,
        vector: &[f32],
        n: usize,
        search_k: i32,
    ) -> anyhow::Result<(Vec<u32>, Vec<f32>)> {
        anyhow::ensure!(vector.len() == self.dimension);
        unsafe {
            let mut results = Vec::with_capacity(n);
            let mut distances = Vec::with_capacity(n);
            let num_results = annoy_angular_get_nns_by_vector(
                self.ptr,
                vector.as_ptr() as *mut _,
                n,
                search_k,
                results.as_mut_ptr(),
                distances.as_mut_ptr(),
            );
            results.set_len(num_results);
            distances.set_len(num_results);
            Ok((results, distances))
        }
    }

    // a.get_item_vector(i) returns the vector for item i that was previously added.
    pub fn get_item_vector(&mut self, item: u32) -> Vec<f32> {
        let mut vector = Vec::with_capacity(self.dimension);
        unsafe {
            annoy_angular_get_item(self.ptr, item, vector.as_mut_ptr());
            vector.set_len(self.dimension);
        }
        vector
    }

    // a.get_distance(i, j) returns the distance between items i and j.
    pub fn get_distance(&mut self, i: u32, j: u32) -> f32 {
        unsafe { annoy_angular_get_distance(self.ptr, i, j) }
    }

    // a.get_n_items() returns the number of items in the index.
    pub fn get_n_items(&mut self) -> u32 {
        unsafe { annoy_angular_get_n_items(self.ptr) }
    }

    // a.on_disk_build(fn) prepares annoy to build the index in the specified file instead
    // of RAM (execute before adding items, no need to save after build)
    pub fn on_disk_build(&mut self, p: &Path) -> anyhow::Result<()> {
        let p_str = p
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("Path {p:?} is not valid UTF-8"))?;
        let p_cstr = CString::new(p_str)?;
        unsafe {
            let mut error_ptr: *mut c_char = ptr::null_mut();
            let success = annoy_angular_on_disk_build(
                self.ptr,
                p_cstr.as_ptr() as *mut _,
                &mut error_ptr as *mut _,
            );
            check_error("on_disk_build", success, error_ptr)?;
        }
        Ok(())
    }
}

fn check_error(name: &str, success: bool, error_ptr: *mut c_char) -> anyhow::Result<()> {
    if success {
        return Ok(());
    }
    if error_ptr.is_null() {
        anyhow::bail!("{name} failed: <unknown error>");
    }
    let message = unsafe { CStr::from_ptr(error_ptr).to_str()? };
    let error = anyhow::anyhow!("{name} failed: {message}");
    unsafe {
        annoy_angular_free_error(error_ptr);
    }
    Err(error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple() -> anyhow::Result<()> {
        let mut a = AnnoyAngular::new(3);
        a.add_item(0, &[1.0, 0.0, 0.0])?;
        a.add_item(1, &[0.0, 1.0, 0.0])?;
        a.add_item(2, &[0.0, 0.0, 1.0])?;
        a.build(-1)?;

        let (results, distance) = a.get_nearest_by_item(0, 100, -1)?;
        for (r, d) in results.iter().zip(distance.iter()) {
            println!("{} {}", r, d);
        }

        let (results, distance) = a.get_nearest_by_vector(&[1.0, 0.5, 0.5], 100, -1)?;
        for (r, d) in results.iter().zip(distance.iter()) {
            println!("{} {}", r, d);
        }
        Ok(())
    }
}
