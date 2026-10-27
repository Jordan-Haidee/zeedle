fn main() {
    println!("cargo:rerun-if-changed=ui/");
    println!("cargo:rerun-if-changed=lang/");

    let libraries = std::collections::HashMap::from([(
        "lucide".to_string(),
        std::path::PathBuf::from(lucide_slint::lib()),
    )]);
    let cfg = slint_build::CompilerConfiguration::new()
        .with_style("fluent".into())
        .with_bundled_translations("lang")
        .with_library_paths(libraries);
    slint_build::compile_with_config("ui/app.slint", cfg).expect("slint build failed");
    #[cfg(target_os = "windows")]
    {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("ui/cover.ico");
        res.compile().expect("can't use this icon!");
    }
}
