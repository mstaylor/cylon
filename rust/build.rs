// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Build script for generating UCX and UCC FFI bindings
//!
//! Environment variables:
//! - `UCX_INSTALL_PREFIX` - Path to UCX installation (e.g., $HOME/ucx)
//! - `UCC_INSTALL_PREFIX` - Path to UCC installation (e.g., $HOME/ucc)
//!
//! Alternatively, you can use:
//! - `UCX_INCLUDEDIR` and `UCX_LIBDIR` for UCX
//! - `UCC_INCLUDEDIR` and `UCC_LIBDIR` for UCC

use std::env;
use std::path::PathBuf;

fn main() {
    // Re-run if environment variables change
    println!("cargo:rerun-if-env-changed=UCX_INSTALL_PREFIX");
    println!("cargo:rerun-if-env-changed=UCX_INCLUDEDIR");
    println!("cargo:rerun-if-env-changed=UCX_LIBDIR");
    println!("cargo:rerun-if-env-changed=UCC_INSTALL_PREFIX");
    println!("cargo:rerun-if-env-changed=UCC_INCLUDEDIR");
    println!("cargo:rerun-if-env-changed=UCC_LIBDIR");
    println!("cargo:rerun-if-env-changed=CONDA_PREFIX");

    #[cfg(feature = "ucx")]
    build_ucx();

    #[cfg(feature = "ucc")]
    build_ucc();
}

#[cfg(feature = "ucx")]
fn build_ucx() {
    let (include_dir, lib_dir) = get_ucx_paths();

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=ucp");
    println!("cargo:rustc-link-lib=dylib=uct");
    println!("cargo:rustc-link-lib=dylib=ucs");
    println!("cargo:rustc-link-lib=dylib=ucm");

    // Add rpath for runtime linking
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());

    generate_ucx_bindings(&include_dir);
}

#[cfg(feature = "ucc")]
fn build_ucc() {
    let (include_dir, lib_dir) = get_ucc_paths();

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=ucc");

    // Add rpath for runtime linking
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());

    generate_ucc_bindings(&include_dir);
}

#[cfg(feature = "ucx")]
fn get_ucx_paths() -> (PathBuf, PathBuf) {
    // First, try UCX_INSTALL_PREFIX
    if let Ok(install_prefix) = env::var("UCX_INSTALL_PREFIX") {
        let include_dir = PathBuf::from(&install_prefix).join("include");
        let lib_dir = PathBuf::from(&install_prefix).join("lib");

        if !include_dir.exists() || !lib_dir.exists() {
            panic!(
                "UCX_INSTALL_PREFIX is set to {}, but include or lib directories don't exist",
                install_prefix
            );
        }

        return (include_dir, lib_dir);
    }

    // Next, try UCX_INCLUDEDIR and UCX_LIBDIR
    let include_dir = env::var("UCX_INCLUDEDIR")
        .map(PathBuf::from)
        .or_else(|_| {
            // Try conda environment
            env::var("CONDA_PREFIX").map(|p| PathBuf::from(p).join("include"))
        })
        .expect("UCX_INSTALL_PREFIX, UCX_INCLUDEDIR, or CONDA_PREFIX must be set for UCX feature");

    let lib_dir = env::var("UCX_LIBDIR")
        .map(PathBuf::from)
        .or_else(|_| {
            // Try conda environment
            env::var("CONDA_PREFIX").map(|p| PathBuf::from(p).join("lib"))
        })
        .expect("UCX_INSTALL_PREFIX, UCX_LIBDIR, or CONDA_PREFIX must be set for UCX feature");

    if !include_dir.exists() {
        panic!("UCX include directory does not exist: {}", include_dir.display());
    }
    if !lib_dir.exists() {
        panic!("UCX lib directory does not exist: {}", lib_dir.display());
    }

    (include_dir, lib_dir)
}

#[cfg(feature = "ucc")]
fn get_ucc_paths() -> (PathBuf, PathBuf) {
    // First, try UCC_INSTALL_PREFIX
    if let Ok(install_prefix) = env::var("UCC_INSTALL_PREFIX") {
        let include_dir = PathBuf::from(&install_prefix).join("include");
        let lib_dir = PathBuf::from(&install_prefix).join("lib");

        if !include_dir.exists() || !lib_dir.exists() {
            panic!(
                "UCC_INSTALL_PREFIX is set to {}, but include or lib directories don't exist",
                install_prefix
            );
        }

        return (include_dir, lib_dir);
    }

    // Next, try UCC_INCLUDEDIR and UCC_LIBDIR
    let include_dir = env::var("UCC_INCLUDEDIR")
        .map(PathBuf::from)
        .or_else(|_| {
            // Try conda environment
            env::var("CONDA_PREFIX").map(|p| PathBuf::from(p).join("include"))
        })
        .expect("UCC_INSTALL_PREFIX, UCC_INCLUDEDIR, or CONDA_PREFIX must be set for UCC feature");

    let lib_dir = env::var("UCC_LIBDIR")
        .map(PathBuf::from)
        .or_else(|_| {
            // Try conda environment
            env::var("CONDA_PREFIX").map(|p| PathBuf::from(p).join("lib"))
        })
        .expect("UCC_INSTALL_PREFIX, UCC_LIBDIR, or CONDA_PREFIX must be set for UCC feature");

    if !include_dir.exists() {
        panic!("UCC include directory does not exist: {}", include_dir.display());
    }
    if !lib_dir.exists() {
        panic!("UCC lib directory does not exist: {}", lib_dir.display());
    }

    (include_dir, lib_dir)
}

#[cfg(feature = "ucx")]
fn generate_ucx_bindings(include_dir: &PathBuf) {
    let bindings = bindgen::Builder::default()
        .header_contents(
            "ucx_wrapper.h",
            r#"
            #include <ucp/api/ucp.h>
            #include <uct/api/uct.h>
            #include <ucs/type/status.h>
            "#,
        )
        .clang_arg(format!("-I{}", include_dir.display()))
        // Generate bindings for UCX types and functions
        .allowlist_type("ucp_.*")
        .allowlist_type("uct_.*")
        .allowlist_type("ucs_.*")
        .allowlist_function("ucp_.*")
        .allowlist_function("uct_.*")
        .allowlist_function("ucs_.*")
        .allowlist_var("UCP_.*")
        .allowlist_var("UCT_.*")
        .allowlist_var("UCS_.*")
        // Derive common traits
        .derive_debug(true)
        .derive_default(true)
        .derive_partialeq(true)
        // Generate correct types
        .size_t_is_usize(true)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate UCX bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("ucx_bindings.rs"))
        .expect("Couldn't write UCX bindings!");

    println!("cargo:warning=Generated UCX bindings at {}/ucx_bindings.rs", out_path.display());
}

#[cfg(feature = "ucc")]
fn generate_ucc_bindings(include_dir: &PathBuf) {
    let bindings = bindgen::Builder::default()
        .header_contents(
            "ucc_wrapper.h",
            r#"
            #include <ucc/api/ucc.h>
            "#,
        )
        .clang_arg(format!("-I{}", include_dir.display()))
        // Generate bindings for UCC types and functions
        .allowlist_type("ucc_.*")
        .allowlist_function("ucc_.*")
        .allowlist_var("UCC_.*")
        // Derive common traits
        .derive_debug(true)
        .derive_default(true)
        .derive_partialeq(true)
        // Generate correct types
        .size_t_is_usize(true)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate UCC bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("ucc_bindings.rs"))
        .expect("Couldn't write UCC bindings!");

    println!("cargo:warning=Generated UCC bindings at {}/ucc_bindings.rs", out_path.display());
}

// Stub functions for when features are not enabled
#[cfg(not(feature = "ucx"))]
fn build_ucx() {}

#[cfg(not(feature = "ucc"))]
fn build_ucc() {}
