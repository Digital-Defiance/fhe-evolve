extern crate napi_build;

fn main() {
    napi_build::setup();

    // Enable aarch64_neon cfg flag on Apple Silicon targets
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if target_arch == "aarch64" {
        println!("cargo:rustc-cfg=aarch64_neon");
    }
}
