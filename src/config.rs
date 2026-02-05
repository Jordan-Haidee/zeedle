use std::path::PathBuf;

use crate::slint_types::{PlayMode, SortKey};

/// Get config file path
fn get_cfg_path() -> PathBuf {
    home::home_dir().expect("no home directory found").join(".config/zeedle/config.toml")
}

/// Used to save/recover ui state
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct Config {
    pub song_dir: PathBuf,
    pub current_song_path: Option<PathBuf>,
    pub progress: f32,
    pub play_mode: PlayMode,
    pub sort_key: SortKey,
    pub sort_ascending: bool,
    pub lang: String,
    pub light_ui: bool,
    pub volume: f32,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            song_dir: home::home_dir().expect("no home directory found").join("Music"),
            current_song_path: None,
            progress: 0.0,
            play_mode: PlayMode::InOrder,
            sort_key: SortKey::BySongName,
            sort_ascending: true,
            lang: "".into(),
            light_ui: false,
            volume: 1f32,
        }
    }
}

impl Config {
    /// Load config from file, or return default if file not exists or invalid
    pub fn load() -> Self {
        let cfg_path = get_cfg_path();
        if cfg_path.exists() {
            let content = std::fs::read_to_string(&cfg_path).expect("failed to read config file");
            toml::from_str(&content).unwrap_or_default()
        } else {
            Self::default()
        }
    }

    /// Save config to file
    pub fn save(self) {
        let cfg_path = get_cfg_path();
        if let Some(parent) = cfg_path.parent() {
            std::fs::create_dir_all(parent).expect("failed to create config directory");
        }
        let content = toml::to_string_pretty(&self).expect("failed to serialize config");
        std::fs::write(cfg_path, content).expect("failed to write config file");
    }
}
