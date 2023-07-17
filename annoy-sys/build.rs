use std::{env, path::PathBuf};

use bindgen::{Builder, CargoCallbacks};

fn main() {
    cc::Build::new()
        .cpp(true)
        .file("wrapper.cpp")
        .flag("-std=c++14")
        .cpp_link_stdlib("c++")
        .warnings(false)
        .compile("libannoy.a");

    println!("cargo:rerun-if-changed=wrapper.hpp");
    let bindings = Builder::default()
        .clang_arg("-xc++")
        .header("wrapper.hpp")
        .allowlist_function("annoy_angular_.*")
        .parse_callbacks(Box::new(CargoCallbacks))
        .generate()
        .expect("Failed to generate bindings");
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Failed to write bindings");
}
