fn main() {
    println!("cargo:rerun-if-changed=ui/");
    println!("cargo:rerun-if-changed=lang/");

    let cfg = slint_build::CompilerConfiguration::new()
        .with_style("fluent".into())
        .with_bundled_translations("lang");
    slint_build::compile_with_config("ui/app.slint", cfg).expect("slint build failed");
    #[cfg(target_os = "windows")]
    {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("ui/cover.ico");
        res.compile().expect("can't use this icon!");
    }
}
