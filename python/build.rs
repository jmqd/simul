//! Passes macOS dynamic-lookup linker flags for Python `cdylib` extensions so
//! `cargo build --manifest-path python/Cargo.toml` works from any directory.

fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-cdylib-link-arg=-undefined");
        println!("cargo:rustc-cdylib-link-arg=dynamic_lookup");
    }
}
