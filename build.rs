fn main() {
    println!("cargo:rerun-if-changed=ui/app.slint");
    println!("cargo:rerun-if-changed=lang/zeedle.pot");
    println!("cargo:rerun-if-changed=lang/de/LC_MESSAGES/zeedle.po");
    println!("cargo:rerun-if-changed=lang/es/LC_MESSAGES/zeedle.po");
    println!("cargo:rerun-if-changed=lang/fr/LC_MESSAGES/zeedle.po");
    println!("cargo:rerun-if-changed=lang/ru/LC_MESSAGES/zeedle.po");
    println!("cargo:rerun-if-changed=lang/zh_CN/LC_MESSAGES/zeedle.po");

    let cfg = slint_build::CompilerConfiguration::new()
        .with_style("fluent".into())
        .with_bundled_translations("lang");
    slint_build::compile_with_config("ui/app.slint", cfg).expect("slint build failed");
    if std::env::var("CARGO_CFG_TARGET_OS").expect("can't find this env variable!") == "windows" {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("ui/cover.ico");
        res.compile().expect("can't use this icon!");
    }
}
