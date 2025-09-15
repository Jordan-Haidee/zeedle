fn main() {
    let cfg = slint_build::CompilerConfiguration::new().with_style("fluent-dark".into());
    slint_build::compile_with_config("ui/app.slint", cfg).expect("slint build failed");
    if std::env::var("CARGO_CFG_TARGET_OS").expect("can't find this env variable!") == "windows" {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("ui/cover.ico");
        res.compile().expect("can't use this icon!");
    }
}
