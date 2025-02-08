fn main() {
    let target_os = "macos".to_string();//std::env::var("CARGO_CFG_TARGET_OS").unwrap();
    let target_arch = "aarch64".to_string();//std::env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    
    

    if target_os == "macos" && target_arch == "aarch64" {
        println!("cargo:rustc-cfg=feature=\"metal\"");
        println!("cargo:rustc-cfg=metal");
    } else {
        println!("cargo:rustc-cfg=feature=\"cuda\"");
        println!("cargo:rustc-cfg=cuda");
    }
}
