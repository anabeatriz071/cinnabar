//! Embeds externally compiled, hash-pinned native shader libraries in Windows previews.
fn main() {
    println!("cargo::rustc-check-cfg=cfg(native_dxr_shader)");
    println!("cargo::rerun-if-env-changed=CINNABAR_NATIVE_DXR_DXIL");
    let Some(path) = std::env::var_os("CINNABAR_NATIVE_DXR_DXIL") else {
        return;
    };
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    println!(
        "cargo::rerun-if-changed={}",
        std::path::Path::new(&path).display()
    );
    let bytes =
        std::fs::read(path).expect("compile native terrain shaders before building the preview");
    assert!(
        bytes.starts_with(b"DXBC") && bytes.len() > 32,
        "invalid native DXIL container"
    );
    let output =
        std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap()).join("native-terrain.dxil");
    std::fs::write(output, bytes).unwrap();
    println!("cargo::rustc-cfg=native_dxr_shader");
}
