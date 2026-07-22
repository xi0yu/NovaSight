fn main() {
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_DEEPSTREAM");
    if cfg!(target_os = "linux") && std::env::var_os("CARGO_FEATURE_DEEPSTREAM").is_some() {
        // Production layout: /opt/novasight/bin/novasightd and
        // /opt/novasight/lib/libnovasight_deepstream_bridge.so.
        println!("cargo:rustc-link-arg-bin=novasightd=-Wl,-rpath,$ORIGIN/../lib");
    }
}
