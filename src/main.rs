#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
use std::{
    cell::Cell,
    cmp::Reverse,
    path::PathBuf,
    rc::Rc,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};

use rand::Rng;
use rayon::slice::ParallelSliceMut;
use rodio::{Decoder, Source};
use slint::{Model, ToSharedString, winit_030::WinitWindowAccessor};

mod slint_types;
use slint_types::*;
mod config;
use config::Config;
mod logger;
mod spectrum;
use spectrum::{
    SPECTRUM_UPDATE_MS, SpectrumChunk, TapSource, default_spectrum, try_start_spectrum_worker,
};
fn is_system_light() -> bool {
    matches!(dark_light::detect(), Ok(dark_light::Mode::Light))
}

mod utils;

/// Message in channel: ui --> backend
/// Note: messages in the opposite direction (backend --> ui) are sent via slint::invoke_from_event_loop
enum PlayerCommand {
    Play(SongInfo, TriggerSource), // 从头播放某个音频文件
    Pause,                         // 暂停/继续播放
    ChangeProgress(f32),           // 拖拽进度条
    PlayNext,                      // 播放下一首
    PlayPrev,                      // 播放上一首
    SwitchMode(PlayMode),          // 切换播放模式
    RefreshSongList(PathBuf),      // 刷新歌曲列表
    SortSongList(SortKey, bool),   // 刷新歌曲列表
    SetLang(String),               // 设置语言
    ChangeVolume(f32),             // 改变音量
    SetShowSpectrum(bool),         // 设置是否显示频谱
}

/// Apply config values that don't depend on a current song.
/// Used for both empty-list startup and normal startup.
fn apply_config_to_ui(ui: &MainWindow, cfg: &Config) {
    let settings = ui.global::<SettingsGlobal>();
    settings.set_sort_key(cfg.sort_key);
    settings.set_sort_ascending(cfg.sort_ascending);
    settings.set_last_sort_key(cfg.sort_key);
    settings.set_lang(cfg.lang.clone().into());
    settings.set_song_dir(cfg.song_dir.to_str().expect("failed to convert Path to String").into());
    settings.set_about_info(utils::get_about_info());
    settings.set_follow_system_theme(cfg.follow_system_theme);

    let playback = ui.global::<PlaybackGlobal>();
    playback.set_paused(true);
    playback.set_play_mode(cfg.play_mode);
    playback.set_volume(cfg.volume);
    playback.set_show_spectrum(cfg.show_spectrum);

    slint::select_bundled_translation(&cfg.lang)
        .unwrap_or_else(|_| panic!("failed to set language: {}", cfg.lang));
    if cfg.follow_system_theme {
        let is_light = is_system_light();
        ui.invoke_set_light_theme(is_light);
        log::info!(
            "system theme detected: {}",
            if is_light {
                "light"
            } else {
                "dark"
            }
        );
    } else {
        ui.invoke_set_light_theme(cfg.light_ui);
    }
}

/// Clear song-dependent fields only. Keeps user settings (volume, lang,
/// play_mode, show_spectrum, song_dir, theme) intact.
fn clear_song_state(ui: &MainWindow) {
    let playback = ui.global::<PlaybackGlobal>();
    playback.set_progress(0.0);
    playback.set_duration(0.0);
    playback.set_spectrum(default_spectrum().as_slice().into());
    playback.set_paused(true);
    playback.set_dragging(false);
    playback.set_user_listening(false);

    let now_playing = ui.global::<NowPlayingGlobal>();
    now_playing.set_album_image(
        slint::Image::load_from_svg_data(include_bytes!("../ui/cover.svg"))
            .expect("failed to load default image"),
    );
    now_playing.set_current_song(SongInfo {
        id: -1,
        song_path: "".into(),
        song_name: "No song".into(),
        singer: "unknown".into(),
        duration: "00:00".into(),
    });
    now_playing.set_lyrics(Vec::new().as_slice().into());
    now_playing.set_lyric_viewport_y(0.);

    let settings = ui.global::<SettingsGlobal>();
    settings.set_song_list(Vec::new().as_slice().into());
}

/// Set UI state according to saved config
fn set_start_ui_state(ui: &MainWindow, cfg: &Config) -> Option<(SongInfo, f32, f32, bool)> {
    let settings = ui.global::<SettingsGlobal>();
    let app_font = utils::get_default_font_family();
    settings.set_app_font_family(app_font.into());
    let song_list = utils::read_song_list(&cfg.song_dir, cfg.sort_key, cfg.sort_ascending);
    if song_list.is_empty() {
        log::warn!("song list is empty in directory: {:?}, keeping saved config ...", cfg.song_dir);
        apply_config_to_ui(ui, cfg);
        return None;
    }
    log::info!("loaded {} songs from directory: {:?}", song_list.len(), cfg.song_dir);
    apply_config_to_ui(ui, cfg);

    let playback = ui.global::<PlaybackGlobal>();
    playback.set_progress(cfg.progress);
    let now_playing = ui.global::<NowPlayingGlobal>();

    settings.set_song_list(song_list.as_slice().into());
    // Use saved song path if valid, otherwise fall back to first song in list
    let saved_path =
        cfg.current_song_path.as_ref().filter(|p| !p.as_os_str().is_empty() && p.exists());

    // Try saved path first, then iterate through songs as fallback
    let cur_song_info = saved_path
        .and_then(|p| {
            utils::read_meta_info(p).map_or_else(
                |e| {
                    log::warn!("{}", e);
                    None
                },
                Some,
            )
        })
        .or_else(|| {
            utils::read_meta_info(song_list[0].song_path.as_str()).map_or_else(
                |e| {
                    log::warn!("{}", e);
                    None
                },
                Some,
            )
        })
        .or_else(|| {
            song_list.iter().find_map(|s| {
                utils::read_meta_info(&s.song_path).map_or_else(
                    |e| {
                        log::warn!("{}", e);
                        None
                    },
                    Some,
                )
            })
        });

    let mut cur_song_info = match cur_song_info {
        Some(info) => info,
        None => {
            log::error!("all songs failed to load, clearing song state");
            clear_song_state(ui);
            return None;
        }
    };
    cur_song_info.id = song_list
        .iter()
        .find(|x| x.song_path == cur_song_info.song_path)
        .map(|x| x.id)
        .unwrap_or(0);
    let dura = cur_song_info
        .clone()
        .duration
        .split(':')
        .map(|x| x.parse::<f32>().unwrap_or(0.))
        .rev()
        .reduce(|acc, x| acc + x * 60.)
        .unwrap_or(0.);
    playback.set_duration(dura);
    now_playing.set_current_song(cur_song_info.clone());
    let cover = utils::read_album_cover(&cur_song_info.song_path);
    let cover = match cover {
        Some((buffer, width, height)) => {
            let mut pixel_buffer = slint::SharedPixelBuffer::new(width, height);
            let pixel_buffer_data = pixel_buffer.make_mut_bytes();
            pixel_buffer_data.copy_from_slice(&buffer);
            slint::Image::from_rgba8(pixel_buffer)
        }
        None => utils::get_default_album_cover(),
    };
    now_playing.set_album_image(cover);
    now_playing.set_lyrics(utils::read_lyrics(&cur_song_info.song_path).as_slice().into());
    playback.set_spectrum(default_spectrum().as_slice().into());
    let mut history = now_playing.get_play_history().iter().collect::<Vec<_>>();
    history.push(cur_song_info.clone());
    now_playing.set_play_history(history.as_slice().into());
    now_playing.set_history_index(0);
    Some((cur_song_info, cfg.volume, cfg.progress, cfg.show_spectrum))
}

fn set_start_player_state(
    player: &rodio::Player,
    cur_song_info: &SongInfo,
    volume: f32,
    progress: f32,
    spectrum_enabled: Arc<AtomicBool>,
    fft_tx: &mpsc::SyncSender<SpectrumChunk>,
) {
    let file = std::fs::File::open(&cur_song_info.song_path)
        .unwrap_or_else(|_| panic!("failed to open audio file: {}", cur_song_info.song_path));
    let source = Decoder::try_from(file).expect("failed to decode audio file");
    let tap = TapSource::new(source, fft_tx.clone(), spectrum_enabled.clone());
    player.append(tap);
    player.pause();
    player.set_volume(volume);
    player.try_seek(Duration::from_secs_f32(progress)).expect("failed to seek to given position");
}

fn start_player_backend_thread(
    rx: mpsc::Receiver<PlayerCommand>,
    ui_weak: slint::Weak<MainWindow>,
    player_clone: Arc<Mutex<rodio::Player>>,
    fft_tx: mpsc::SyncSender<SpectrumChunk>,
    spectrum_enabled: Arc<AtomicBool>,
) {
    thread::spawn(move || {
        log::info!("player thread running...");
        while let Ok(cmd) = rx.recv() {
            match cmd {
                PlayerCommand::Play(song_info, trigger) => {
                    let file = std::fs::File::open(&song_info.song_path)
                        .expect("failed to open audio file");
                    let source = Decoder::try_from(file).expect("failed to decode audio file");
                    let dura = source.total_duration().map(|d| d.as_secs_f32()).unwrap_or(0.0);
                    let lyrics = utils::read_lyrics(&song_info.song_path);
                    let player_guard = player_clone.lock().unwrap();
                    player_guard.clear();
                    let tap = TapSource::new(source, fft_tx.clone(), spectrum_enabled.clone());
                    player_guard.append(tap);
                    player_guard.play();
                    log::info!("start playing: <{}>", song_info.song_name);
                    let cover = utils::read_album_cover(&song_info.song_path);
                    let ui_weak = ui_weak.clone();
                    slint::invoke_from_event_loop(move || {
                        if let Some(ui) = ui_weak.upgrade() {
                            let now_playing = ui.global::<NowPlayingGlobal>();
                            match trigger {
                                TriggerSource::ClickItem => {
                                    let mut history =
                                        now_playing.get_play_history().iter().collect::<Vec<_>>();
                                    history.push(song_info.clone());
                                    now_playing.set_play_history(history.as_slice().into());
                                    now_playing.set_history_index(0);
                                }
                                TriggerSource::Prev => {
                                    let history =
                                        now_playing.get_play_history().iter().collect::<Vec<_>>();
                                    let new_index = now_playing.get_history_index() + 1;
                                    now_playing
                                        .set_history_index(new_index.min(history.len() as i32 - 1));
                                }
                                TriggerSource::Next => {
                                    if now_playing.get_history_index() > 0 {
                                        now_playing
                                            .set_history_index(now_playing.get_history_index() - 1);
                                    } else {
                                        if ui.global::<PlaybackGlobal>().get_play_mode()
                                            != PlayMode::Recursive
                                        {
                                            let mut history = now_playing
                                                .get_play_history()
                                                .iter()
                                                .collect::<Vec<_>>();
                                            history.push(song_info.clone());
                                            now_playing.set_play_history(history.as_slice().into());
                                        }
                                        now_playing.set_history_index(0);
                                    }
                                }
                            }

                            now_playing.set_current_song(song_info.clone());
                            now_playing.set_lyrics(lyrics.as_slice().into());
                            now_playing.set_lyric_viewport_y(0.);

                            let playback = ui.global::<PlaybackGlobal>();
                            playback.set_paused(false);
                            playback.set_progress(0.0);
                            playback.set_duration(dura);
                            playback.set_user_listening(true);

                            let cover = match cover {
                                Some((buffer, width, height)) => {
                                    let mut pixel_buffer =
                                        slint::SharedPixelBuffer::new(width, height);
                                    let pixel_buffer_data = pixel_buffer.make_mut_bytes();
                                    pixel_buffer_data.copy_from_slice(&buffer);
                                    slint::Image::from_rgba8(pixel_buffer)
                                }
                                None => utils::get_default_album_cover(),
                            };
                            now_playing.set_album_image(cover);

                            log::debug!(
                                "{:?} / {}",
                                now_playing
                                    .get_play_history()
                                    .iter()
                                    .map(|x| x.id)
                                    .collect::<Vec<_>>(),
                                now_playing.get_history_index()
                            );
                        }
                    })
                    .unwrap();
                }
                PlayerCommand::Pause => {
                    let player_guard = player_clone.lock().unwrap();
                    let ui_weak = ui_weak.clone();
                    if player_guard.empty() {
                        log::info!("Queue is empty, playing the first song in the list");
                        slint::invoke_from_event_loop(move || {
                            if let Some(ui) = ui_weak.upgrade() {
                                let settings = ui.global::<SettingsGlobal>();
                                if let Some(song) = settings.get_song_list().iter().next() {
                                    ui.invoke_play(song.clone(), TriggerSource::ClickItem);
                                    ui.global::<PlaybackGlobal>().set_paused(false);
                                } else {
                                    log::warn!("song list is empty, can't play");
                                }
                            }
                        })
                        .unwrap();
                    } else {
                        let paused = player_guard.is_paused();
                        if paused {
                            player_guard.play();
                        } else {
                            player_guard.pause();
                        }
                        slint::invoke_from_event_loop(move || {
                            if let Some(ui) = ui_weak.upgrade() {
                                let playback = ui.global::<PlaybackGlobal>();
                                playback.set_paused(!paused);
                                playback.set_user_listening(true);
                            }
                        })
                        .unwrap();
                        log::info!("pause/play toggled");
                    }
                }
                PlayerCommand::ChangeProgress(new_progress) => {
                    let player_guard = player_clone.lock().unwrap();
                    match player_guard.try_seek(Duration::from_secs_f32(new_progress)) {
                        Ok(_) => {
                            let ui_weak = ui_weak.clone();
                            slint::invoke_from_event_loop(move || {
                                if let Some(ui) = ui_weak.upgrade() {
                                    ui.global::<PlaybackGlobal>().set_progress(new_progress);
                                }
                            })
                            .unwrap();
                        }
                        Err(e) => {
                            log::error!("Failed to seek: <{}>", e);
                        }
                    }
                }
                PlayerCommand::PlayNext => {
                    let ui_weak = ui_weak.clone();
                    slint::invoke_from_event_loop(move || {
                        if let Some(ui) = ui_weak.upgrade() {
                            let now_playing = ui.global::<NowPlayingGlobal>();
                            if now_playing.get_history_index() > 0 {
                                // 如果处在历史播放模式，则先尝试从历史记录中获取下一首
                                log::info!("playing next from history");
                                let history =
                                    now_playing.get_play_history().iter().collect::<Vec<_>>();
                                if let Some(song) = history
                                    .iter()
                                    .rev()
                                    .nth((now_playing.get_history_index() - 1) as usize)
                                {
                                    ui.invoke_play(song.clone(), TriggerSource::Next);
                                } else {
                                    log::warn!("failed to play next song in history");
                                }
                            } else {
                                // 否则根据播放模式获取下一首
                                log::info!("playing next from play mode");
                                let settings = ui.global::<SettingsGlobal>();
                                let playback = ui.global::<PlaybackGlobal>();
                                let song_list: Vec<_> = settings.get_song_list().iter().collect();
                                if song_list.is_empty() {
                                    log::warn!("song list is empty, can't play next");
                                    return;
                                }
                                let mut rng = rand::rng();
                                let next_id1 = rng.random_range(..song_list.len());
                                let id = now_playing.get_current_song().id as usize;
                                let mut next_id2 = if id + 1 >= song_list.len() {
                                    0
                                } else {
                                    id + 1
                                };
                                next_id2 = next_id2.min(song_list.len() - 1);
                                let next_id = match playback.get_play_mode() {
                                    PlayMode::InOrder => next_id2,
                                    PlayMode::Random => next_id1,
                                    PlayMode::Recursive => id,
                                };
                                if let Some(next_song) = song_list.get(next_id) {
                                    let song_to_play = next_song.clone();
                                    ui.invoke_play(song_to_play.clone(), TriggerSource::Next);
                                } else {
                                    log::warn!("failed to play next from play mode");
                                }
                            }
                        }
                    })
                    .unwrap();
                }
                PlayerCommand::PlayPrev => {
                    let ui_weak: slint::Weak<MainWindow> = ui_weak.clone();
                    slint::invoke_from_event_loop(move || {
                        if let Some(ui) = ui_weak.upgrade() {
                            let now_playing = ui.global::<NowPlayingGlobal>();
                            let settings = ui.global::<SettingsGlobal>();
                            let cur_song = now_playing.get_current_song();
                            let song_list: Vec<_> = settings.get_song_list().iter().collect();
                            if song_list.is_empty() {
                                log::warn!("song list is empty, can't play prev");
                                return;
                            }
                            let history = now_playing.get_play_history().iter().collect::<Vec<_>>();
                            if let Some(song) = history
                                .iter()
                                .rev()
                                .nth((now_playing.get_history_index() + 1) as usize)
                            {
                                ui.invoke_play(song.clone(), TriggerSource::Prev);
                                log::info!("playing prev from history");
                            } else {
                                ui.invoke_play(cur_song, TriggerSource::Prev);
                                log::info!("can't get earlier history, fall back to replay oldest history song...");
                            }
                        }
                    })
                    .unwrap();
                }
                PlayerCommand::SwitchMode(m) => {
                    let ui_weak = ui_weak.clone();
                    slint::invoke_from_event_loop(move || {
                        if let Some(ui) = ui_weak.upgrade() {
                            ui.global::<PlaybackGlobal>().set_play_mode(m);
                            log::info!("play mode switched to <{:?}>", m);
                        }
                    })
                    .unwrap();
                }
                PlayerCommand::RefreshSongList(path) => {
                    let new_list = utils::read_song_list(&path, SortKey::BySongName, true);
                    let ui_weak = ui_weak.clone();
                    let player_clone = player_clone.clone();
                    slint::invoke_from_event_loop(move || {
                        if let Some(ui) = ui_weak.upgrade() {
                            let settings = ui.global::<SettingsGlobal>();
                            settings.set_song_list(new_list.as_slice().into());
                            settings.set_sort_key(SortKey::BySongName);
                            settings.set_sort_ascending(true);
                            if let Some(first_song) = new_list.first() {
                                ui.invoke_play(first_song.clone(), TriggerSource::ClickItem);
                            } else {
                                let player_guard = player_clone.lock().unwrap();
                                player_guard.clear();
                                clear_song_state(&ui);
                                log::warn!("song list is empty, cleared song state");
                            }
                        }
                    })
                    .unwrap();
                }
                PlayerCommand::SortSongList(key, ascending) => {
                    let ui_weak = ui_weak.clone();
                    slint::invoke_from_event_loop(move || {
                        if let Some(ui) = ui_weak.upgrade() {
                            let settings = ui.global::<SettingsGlobal>();
                            let now_playing = ui.global::<NowPlayingGlobal>();
                            let mut song_list: Vec<_> = settings.get_song_list().iter().collect();
                            if song_list.is_empty() {
                                log::warn!("song list is empty, can't sort");
                                return;
                            }
                            match key {
                                SortKey::BySongName => {
                                    if ascending {
                                        song_list.par_sort_by_key(|a| {
                                            utils::get_chars(a.song_name.as_str())
                                        })
                                    } else {
                                        song_list.par_sort_by_key(|a| {
                                            Reverse(utils::get_chars(a.song_name.as_str()))
                                        })
                                    }
                                }
                                SortKey::BySinger => {
                                    if ascending {
                                        song_list.par_sort_by_key(|a| {
                                            utils::get_chars(a.singer.as_str())
                                        })
                                    } else {
                                        song_list.par_sort_by_key(|a| {
                                            Reverse(utils::get_chars(a.singer.as_str()))
                                        })
                                    }
                                }
                                SortKey::ByDuration => {
                                    if ascending {
                                        song_list.par_sort_by_key(|a| a.duration.clone());
                                    } else {
                                        song_list.par_sort_by_key(|a| Reverse(a.duration.clone()));
                                    }
                                }
                            }
                            song_list.iter_mut().enumerate().for_each(|(i, x)| x.id = i as i32);
                            let new_cur_song = song_list
                                .iter()
                                .find(|x| x.song_path == now_playing.get_current_song().song_path)
                                .unwrap();
                            now_playing.set_current_song(new_cur_song.clone());
                            settings.set_sort_key(key);
                            settings.set_sort_ascending(ascending);
                            settings.set_last_sort_key(key);
                            settings.set_song_list(song_list.as_slice().into());
                            log::info!("song list sorted by <{:?}>, ascending: {}", key, ascending);
                        }
                    })
                    .unwrap();
                }
                PlayerCommand::SetLang(lang) => {
                    let ui_weak = ui_weak.clone();
                    slint::invoke_from_event_loop(move || {
                        if let Some(ui) = ui_weak.upgrade() {
                            slint::select_bundled_translation(&lang)
                                .expect("failed to set language");
                            ui.global::<SettingsGlobal>().set_lang(lang.into());
                        }
                    })
                    .unwrap()
                }
                PlayerCommand::ChangeVolume(v) => {
                    let player_guard = player_clone.lock().unwrap();
                    player_guard.set_volume(v);
                    log::info!("volume changed to: {}", v);
                    let ui_weak = ui_weak.clone();
                    slint::invoke_from_event_loop(move || {
                        if let Some(ui) = ui_weak.upgrade() {
                            ui.global::<PlaybackGlobal>().set_volume(v);
                        }
                    })
                    .unwrap()
                }
                PlayerCommand::SetShowSpectrum(show) => {
                    spectrum_enabled.store(show, Ordering::Relaxed);
                    let ui_weak = ui_weak.clone();
                    slint::invoke_from_event_loop(move || {
                        if let Some(ui) = ui_weak.upgrade() {
                            let playback = ui.global::<PlaybackGlobal>();
                            playback.set_show_spectrum(show);
                            if !show {
                                playback.set_spectrum(default_spectrum().as_slice().into());
                            }
                        }
                    })
                    .unwrap();
                }
            }
        }
    });
}

fn register_ui_callbacks(ui: &MainWindow, tx: mpsc::Sender<PlayerCommand>) {
    {
        let tx = tx.clone();
        ui.on_play(move |song_info: SongInfo, trigger: TriggerSource| {
            log::info!("request to play: <{}> from source <{:?}>", song_info.song_name, trigger);
            tx.send(PlayerCommand::Play(song_info, trigger)).expect("failed to send play command");
        });
    }
    {
        let tx = tx.clone();
        ui.on_toggle_play(move || {
            log::info!("request to toggle play/pause");
            tx.send(PlayerCommand::Pause).expect("failed to send pause command");
        });
    }
    {
        let tx = tx.clone();
        ui.on_change_progress(move |new_progress: f32| {
            log::info!("request to change progress to: <{}>", new_progress);
            tx.send(PlayerCommand::ChangeProgress(new_progress))
                .expect("failed to send change progress command");
        });
    }
    {
        let tx = tx.clone();
        ui.on_play_next(move || {
            log::info!("request to play next");
            tx.send(PlayerCommand::PlayNext).expect("failed to send play next command");
        });
    }
    {
        let tx = tx.clone();
        ui.on_play_prev(move || {
            log::info!("request to play prev");
            tx.send(PlayerCommand::PlayPrev).expect("failed to send play prev command");
        });
    }
    {
        let tx = tx.clone();
        ui.on_switch_mode(move |play_mode| {
            log::info!("request to switch play mode to: {:?}", play_mode);
            tx.send(PlayerCommand::SwitchMode(play_mode))
                .expect("failed to send switch mode command");
        });
    }
    {
        let tx = tx.clone();
        ui.on_refresh_song_list(move |path| {
            log::info!("request to refresh song list from: {:?}", path);
            tx.send(PlayerCommand::RefreshSongList(path.as_str().into()))
                .expect("failed to send refresh song list command");
        });
    }
    {
        let tx = tx.clone();
        ui.on_sort_song_list(move |key, ascending| {
            log::info!("request to sort song list by: {:?}, ascending: {}", key, ascending);
            tx.send(PlayerCommand::SortSongList(key, ascending))
                .expect("failed to send sort song list command");
        });
    }
    {
        let tx = tx.clone();
        ui.on_set_lang(move |lang| {
            log::info!("request to set language to: {}", lang);
            tx.send(PlayerCommand::SetLang(lang.as_str().into()))
                .expect("failed to send set language command");
        });
    }
    {
        let tx = tx.clone();
        ui.on_change_volume(move |v| {
            log::info!("request to change volume to: {}", v);
            tx.send(PlayerCommand::ChangeVolume(v)).expect("failed to send change volume command");
        });
    }
    {
        let tx = tx.clone();
        ui.on_set_show_spectrum(move |show| {
            log::info!("request to set show_spectrum to: {}", show);
            tx.send(PlayerCommand::SetShowSpectrum(show))
                .expect("failed to send set show_spectrum command");
        });
    }
    tx.send(PlayerCommand::SetShowSpectrum(ui.global::<PlaybackGlobal>().get_show_spectrum()))
        .expect("failed to send initial show_spectrum command");

    {
        let ui_weak = ui.as_weak();
        let tx = tx.clone();
        ui.on_select_song_dir(move || {
            if let Some(path) =
                rfd::FileDialog::new().set_title("Select Music Directory").pick_folder()
            {
                let path_str = path.display().to_string();
                log::info!("music directory selected: {}", path_str);
                if let Some(ui) = ui_weak.upgrade() {
                    ui.global::<SettingsGlobal>().set_song_dir(path_str.as_str().into());
                }
                tx.send(PlayerCommand::RefreshSongList(path))
                    .expect("failed to send refresh song list command");
            }
        });
    }

    {
        let ui_weak = ui.as_weak();
        ui.on_set_follow_system_theme(move |follow| {
            log::info!("request to set follow_system_theme to: {}", follow);
            if let Some(ui) = ui_weak.upgrade() {
                ui.global::<SettingsGlobal>().set_follow_system_theme(follow);
                if follow {
                    let is_light = is_system_light();
                    ui.invoke_set_light_theme(is_light);
                    log::info!(
                        "follow system theme enabled, detected: {}",
                        if is_light {
                            "light"
                        } else {
                            "dark"
                        }
                    );
                }
            }
        });
    }

    // window control callbacks
    let ui_weak = ui.as_weak();
    ui.on_window_minimize(move || {
        if let Some(ui) = ui_weak.upgrade() {
            ui.window().set_minimized(true);
        }
    });

    // Position cache for fallback drag — defined before maximize
    // so the handler can reset it when window state changes.
    let init_pos = Rc::new(Cell::new(None::<(i32, i32)>));

    let ui_weak = ui.as_weak();
    let init_pos_max = init_pos.clone();
    ui.on_window_maximize(move || {
        if let Some(ui) = ui_weak.upgrade() {
            let window = ui.window();
            window.set_maximized(!window.is_maximized());
        }
        // Reset position cache — window position changed, so the
        // cached fallback origin from a previous gesture is stale.
        init_pos_max.set(None);
    });
    let ui_weak = ui.as_weak();
    ui.on_window_close(move || {
        if let Some(ui) = ui_weak.upgrade() {
            save_ui_state(&ui);
            let _ = ui.window().hide();
            slint::quit_event_loop().ok();
        }
    });

    // Window dragging — use native OS move via winit's drag_window()
    let ui_weak = ui.as_weak();
    ui.on_window_drag_delta(move |dx: f32, dy: f32| {
        if let Some(app) = ui_weak.upgrade() {
            // Try native OS drag (works on Wayland, X11, Windows when button is pressed)
            let native_ok =
                app.window().with_winit_window(|w| w.drag_window().is_ok()).unwrap_or(false);

            if native_ok {
                // OS window manager/compositor handles all movement
                init_pos.set(None);
                return;
            }

            // Fallback: manually position the window (non-winit backends)
            let (ix, iy) = init_pos.get().unwrap_or_else(|| {
                let pos = app.window().position();
                init_pos.set(Some((pos.x, pos.y)));
                (pos.x, pos.y)
            });
            app.window().set_position(slint::PhysicalPosition::new(ix + dx as i32, iy + dy as i32));
        }
    });

    // pure callback to format duration string
    ui.on_format_duration(|dura| {
        format!("{:02}:{:02}", (dura as u32) / 60, (dura as u32) % 60).to_shared_string()
    });
}

fn build_progress_timer(
    ui_weak: slint::Weak<MainWindow>,
    player_clone: Arc<Mutex<rodio::Player>>,
) -> slint::Timer {
    let timer = slint::Timer::default();
    timer.start(slint::TimerMode::Repeated, Duration::from_millis(200), move || {
        if let Some(ui) = ui_weak.upgrade() {
            let playback = ui.global::<PlaybackGlobal>();
            let now_playing = ui.global::<NowPlayingGlobal>();
            if playback.get_paused() {
                return;
            }
            let player_guard = player_clone.lock().unwrap();
            // 如果不在拖动进度条，则自增进度条
            if !playback.get_dragging() {
                playback.set_progress(player_guard.get_pos().as_secs_f32());
            }
            for (idx, item) in now_playing.get_lyrics().iter().enumerate() {
                let delta = item.time - playback.get_progress();
                if delta < 0. && delta > -0.20 {
                    if idx <= 5 {
                        now_playing.set_lyric_viewport_y(0.)
                    } else {
                        now_playing.set_lyric_viewport_y(
                            (5_f32 - idx as f32) * now_playing.get_lyric_line_height(),
                        );
                    }
                    log::debug!("lyric changed to: <{:?}>", item);
                    break;
                }
            }
            // 如果播放完毕，且之前是在播放状态，则自动播放下一首
            if player_guard.empty() && playback.get_user_listening() {
                ui.invoke_play_next();
                log::info!("song ended, auto play next");
            }
        }
    });
    timer
}

fn build_spectrum_timer(
    ui_weak: slint::Weak<MainWindow>,
    spectrum_data: Arc<Mutex<Vec<f32>>>,
) -> slint::Timer {
    let spectrum_timer = slint::Timer::default();
    spectrum_timer.start(
        slint::TimerMode::Repeated,
        Duration::from_millis(SPECTRUM_UPDATE_MS),
        move || {
            if let Some(ui) = ui_weak.upgrade() {
                let playback = ui.global::<PlaybackGlobal>();
                if playback.get_paused() || !playback.get_show_spectrum() {
                    return;
                }
                if let Ok(guard) = spectrum_data.lock() {
                    playback.set_spectrum(guard.as_slice().into());
                }
            }
        },
    );
    spectrum_timer
}

fn save_ui_state(ui: &MainWindow) {
    let settings = ui.global::<SettingsGlobal>();
    let playback = ui.global::<PlaybackGlobal>();
    let now_playing = ui.global::<NowPlayingGlobal>();
    Config::save(Config {
        song_dir: settings.get_song_dir().as_str().into(),
        current_song_path: Some(now_playing.get_current_song().song_path.as_str().into()),
        progress: playback.get_progress(),
        play_mode: playback.get_play_mode(),
        sort_key: settings.get_sort_key(),
        sort_ascending: settings.get_sort_ascending(),
        lang: settings.get_lang().into(),
        light_ui: settings.get_light_ui(),
        follow_system_theme: settings.get_follow_system_theme(),
        volume: playback.get_volume(),
        show_spectrum: playback.get_show_spectrum(),
    });
}

fn build_theme_timer(ui_weak: slint::Weak<MainWindow>) -> slint::Timer {
    let timer = slint::Timer::default();
    timer.start(slint::TimerMode::Repeated, Duration::from_secs(1), move || {
        if let Some(ui) = ui_weak.upgrade() {
            let settings = ui.global::<SettingsGlobal>();
            if settings.get_follow_system_theme() {
                let is_light = is_system_light();
                let current = settings.get_light_ui();
                if current != is_light {
                    ui.invoke_set_light_theme(is_light);
                    log::info!(
                        "system theme changed to: {}",
                        if is_light {
                            "light"
                        } else {
                            "dark"
                        }
                    );
                }
            }
        }
    });
    timer
}

fn main() {
    let app_start = Instant::now();

    // when panics happen, auto port errors to log
    std::panic::set_hook(Box::new(|info| {
        log::error!("{}", info);
    }));

    // initialize logger
    logger::init_default_logger(None::<PathBuf>);

    // prevent multiple instances
    let ins = single_instance::SingleInstance::new("Zeedle Music Player").unwrap();
    if !ins.is_single() {
        log::warn!("Vanilla player can only run one instance !");
        return;
    }

    // initialize audio output
    let mut sink_handle =
        rodio::DeviceSinkBuilder::open_default_sink().expect("no output device available");
    sink_handle.log_on_drop(false);
    let _player = rodio::Player::connect_new(sink_handle.mixer());
    let player = Arc::new(Mutex::new(_player));

    // 创建消息通道 ui thread --> backend thread
    let (tx, rx) = mpsc::channel::<PlayerCommand>();

    // 初始化 UI 状态
    let cfg = Config::load();
    let ui = MainWindow::new().expect("failed to create UI");
    let start_state = set_start_ui_state(&ui, &cfg);

    // 初始化播放器状态
    let spectrum_data = Arc::new(Mutex::new(default_spectrum()));
    let spectrum_enabled: Arc<AtomicBool>;
    let (fft_tx, fft_rx) = mpsc::sync_channel::<SpectrumChunk>(4);
    if let Some((cur_song_info, volume, progress, show_spectrum)) = start_state {
        spectrum_enabled = Arc::new(AtomicBool::new(show_spectrum));
        set_start_player_state(
            &player.lock().unwrap(),
            &cur_song_info,
            volume,
            progress,
            spectrum_enabled.clone(),
            &fft_tx,
        );
    } else {
        spectrum_enabled = Arc::new(AtomicBool::new(true));
    }

    // 启动频谱分析线程
    try_start_spectrum_worker(fft_rx, spectrum_data.clone(), spectrum_enabled.clone());

    // 启动播放器线程
    start_player_backend_thread(
        rx,
        ui.as_weak(),
        player.clone(),
        fft_tx.clone(),
        spectrum_enabled.clone(),
    );

    // UI 回调函数定义
    register_ui_callbacks(&ui, tx.clone());

    // UI 定时刷新进度条
    let _progress_timer = build_progress_timer(ui.as_weak(), player.clone());

    // UI 定时刷新频谱
    let _spectrum_timer = build_spectrum_timer(ui.as_weak(), spectrum_data.clone());

    // UI 定时检测系统主题
    let _theme_timer = build_theme_timer(ui.as_weak());

    // 设置 XDG app_id，让 dock 正确关联窗口图标
    #[cfg(target_os = "linux")]
    slint::set_xdg_app_id("Zeedle").expect("failed to set xdg app id");

    // 显示 UI，启动事件循环
    log::info!("ui state initialized, take: {:?}", app_start.elapsed());
    ui.run().expect("failed to run UI");

    // 退出前保存状态
    log::info!("saving config...");
    save_ui_state(&ui);
    log::info!("app exited");
}
