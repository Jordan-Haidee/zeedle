#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
use std::{
    cmp::Reverse,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};

use crossbeam_channel::{Sender, bounded};
use rand::Rng;
use rayon::slice::ParallelSliceMut;
use rodio::{Decoder, Source};
use slint::{Model, ToSharedString};
mod slint_types;
use slint_types::*;
mod config;
use config::Config;
mod logger;
mod spectrum;
use spectrum::{
    SPECTRUM_UPDATE_MS, SpectrumChunk, TapSource, default_spectrum, try_start_spectrum_worker,
};
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

/// Set UI state to default (no song)
fn set_raw_ui_state(ui: &MainWindow) {
    let ui_state = ui.global::<UIState>();
    ui_state.set_progress(0.0);
    ui_state.set_duration(0.0);
    ui_state.set_about_info(utils::get_about_info());
    ui_state.set_spectrum(default_spectrum().as_slice().into());
    ui_state.set_album_image(
        slint::Image::load_from_svg_data(include_bytes!("../ui/cover.svg"))
            .expect("failed to load default image"),
    );
    ui_state.set_current_song(SongInfo {
        id: -1,
        song_path: "".into(),
        song_name: "No song".into(),
        singer: "unknown".into(),
        duration: "00:00".into(),
    });
    ui_state.set_lyrics(Vec::new().as_slice().into());
    ui_state.set_song_list(Vec::new().as_slice().into());
    ui_state.set_song_dir(
        Config::default().song_dir.to_str().expect("failed to convert Path to String").into(),
    );
    ui_state.set_play_mode(PlayMode::InOrder);
    ui_state.set_paused(true);
    ui_state.set_dragging(false);
    ui_state.set_user_listening(false);
    ui_state.set_lyric_viewport_y(0.);
    ui_state.set_volume(1.);
    ui_state.set_show_spectrum(false);
}

/// Set UI state according to saved config
fn set_start_ui_state(ui: &MainWindow, cfg: &Config) -> Option<(SongInfo, f32, f32, bool)> {
    let ui_state = ui.global::<UIState>();
    let app_font = utils::get_default_font_family();
    ui_state.set_app_font_family(app_font.into());
    let song_list = utils::read_song_list(&cfg.song_dir, cfg.sort_key, cfg.sort_ascending);
    if song_list.is_empty() {
        log::warn!(
            "song list is empty in directory: {:?}, using default UI state ...",
            cfg.song_dir
        );
        set_raw_ui_state(ui);
        return None;
    }
    log::info!("loaded {} songs from directory: {:?}", song_list.len(), cfg.song_dir);
    ui.invoke_set_light_theme(cfg.light_ui);
    ui_state.set_sort_key(cfg.sort_key);
    ui_state.set_sort_ascending(cfg.sort_ascending);
    ui_state.set_last_sort_key(cfg.sort_key);
    ui_state.set_progress(cfg.progress);
    ui_state.set_paused(true);
    ui_state.set_play_mode(cfg.play_mode);
    ui_state.set_lang(cfg.lang.clone().into());
    ui_state.set_volume(cfg.volume);
    ui_state.set_show_spectrum(cfg.show_spectrum);
    slint::select_bundled_translation(&cfg.lang)
        .unwrap_or_else(|_| panic!("failed to set language: {}", cfg.lang));
    ui_state.set_song_list(song_list.as_slice().into());
    ui_state.set_song_dir(cfg.song_dir.to_str().expect("failed to convert Path to String").into());
    ui_state.set_about_info(utils::get_about_info());
    let mut cur_song_info = utils::read_meta_info(
        cfg.current_song_path.clone().unwrap_or(song_list[0].song_path.as_str().into()),
    )
    .expect("failed to read meta info of current song");
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
    ui_state.set_duration(dura);
    ui_state.set_current_song(cur_song_info.clone());
    ui_state.set_lyrics(utils::read_lyrics(&cur_song_info.song_path).as_slice().into());
    ui_state.set_spectrum(default_spectrum().as_slice().into());
    let cover = utils::read_album_cover(&cur_song_info.song_path);
    let cover = match cover {
        Some((buffer, width, height)) => utils::from_image_to_slint(buffer, width, height),
        None => utils::get_default_album_cover(),
    };
    ui_state.set_album_image(cover);
    let mut history = ui_state.get_play_history().iter().collect::<Vec<_>>();
    history.push(cur_song_info.clone());
    ui_state.set_play_history(history.as_slice().into());
    ui_state.set_history_index(0);
    Some((cur_song_info, cfg.volume, cfg.progress, cfg.show_spectrum))
}

fn set_start_player_state(
    player: &rodio::Player,
    cur_song_info: &SongInfo,
    volume: f32,
    progress: f32,
    spectrum_enabled: Arc<AtomicBool>,
    fft_tx: &Sender<SpectrumChunk>,
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
    fft_tx: Sender<SpectrumChunk>,
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
                            let ui_state = ui.global::<UIState>();
                            match trigger {
                                TriggerSource::ClickItem => {
                                    let mut history =
                                        ui_state.get_play_history().iter().collect::<Vec<_>>();
                                    history.push(song_info.clone());
                                    ui_state.set_play_history(history.as_slice().into());
                                    ui_state.set_history_index(0);
                                }
                                TriggerSource::Prev => {
                                    let history =
                                        ui_state.get_play_history().iter().collect::<Vec<_>>();
                                    let new_index = ui_state.get_history_index() + 1;
                                    ui_state
                                        .set_history_index(new_index.min(history.len() as i32 - 1));
                                }
                                TriggerSource::Next => {
                                    if ui_state.get_history_index() > 0 {
                                        ui_state
                                            .set_history_index(ui_state.get_history_index() - 1);
                                    } else {
                                        if ui_state.get_play_mode() != PlayMode::Recursive {
                                            let mut history = ui_state
                                                .get_play_history()
                                                .iter()
                                                .collect::<Vec<_>>();
                                            history.push(song_info.clone());
                                            ui_state.set_play_history(history.as_slice().into());
                                        }
                                        ui_state.set_history_index(0);
                                    }
                                }
                            }

                            ui_state.set_current_song(song_info.clone());
                            ui_state.set_paused(false);
                            ui_state.set_progress(0.0);
                            ui_state.set_duration(dura);
                            ui_state.set_user_listening(true);
                            ui_state.set_lyrics(lyrics.as_slice().into());
                            ui_state.set_lyric_viewport_y(0.);
                            let cover = match cover {
                                Some((buffer, width, height)) => {
                                    utils::from_image_to_slint(buffer, width, height)
                                }
                                None => utils::get_default_album_cover(),
                            };
                            ui_state.set_album_image(cover);

                            log::debug!(
                                "{:?} / {}",
                                ui_state
                                    .get_play_history()
                                    .iter()
                                    .map(|x| x.id)
                                    .collect::<Vec<_>>(),
                                ui_state.get_history_index()
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
                                let ui_state = ui.global::<UIState>();
                                if let Some(song) = ui_state.get_song_list().iter().next() {
                                    ui.invoke_play(song.clone(), TriggerSource::ClickItem);
                                    ui_state.set_paused(false);
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
                                let ui_state = ui.global::<UIState>();
                                ui_state.set_paused(!paused);
                                ui_state.set_user_listening(true);
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
                                    let ui_state = ui.global::<UIState>();
                                    ui_state.set_progress(new_progress);
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
                            let ui_state = ui.global::<UIState>();
                            if ui_state.get_history_index() > 0 {
                                // 如果处在历史播放模式，则先尝试从历史记录中获取下一首
                                log::info!("playing next from history");
                                let history =
                                    ui_state.get_play_history().iter().collect::<Vec<_>>();
                                if let Some(song) = history
                                    .iter()
                                    .rev()
                                    .nth((ui_state.get_history_index() - 1) as usize)
                                {
                                    ui.invoke_play(song.clone(), TriggerSource::Next);
                                } else {
                                    log::warn!("failed to play next song in history");
                                }
                            } else {
                                // 否则根据播放模式获取下一首
                                log::info!("playing next from play mode");
                                let song_list: Vec<_> = ui_state.get_song_list().iter().collect();
                                if song_list.is_empty() {
                                    log::warn!("song list is empty, can't play next");
                                    return;
                                }
                                let mut rng = rand::rng();
                                let next_id1 = rng.random_range(..song_list.len());
                                let id = ui_state.get_current_song().id as usize;
                                let mut next_id2 = if id + 1 >= song_list.len() {
                                    0
                                } else {
                                    id + 1
                                };
                                next_id2 = next_id2.min(song_list.len() - 1);
                                let next_id = match ui_state.get_play_mode() {
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
                            let ui_state = ui.global::<UIState>();
                            let cur_song = ui_state.get_current_song();
                            let song_list: Vec<_> = ui_state.get_song_list().iter().collect();
                            if song_list.is_empty() {
                                log::warn!("song list is empty, can't play prev");
                                return;
                            }
                            let history = ui_state.get_play_history().iter().collect::<Vec<_>>();
                            if let Some(song) = history
                                .iter()
                                .rev()
                                .nth((ui_state.get_history_index() + 1) as usize)
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
                            let ui_state = ui.global::<UIState>();
                            ui_state.set_play_mode(m);
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
                            let ui_state = ui.global::<UIState>();
                            ui_state.set_song_list(new_list.as_slice().into());
                            ui_state.set_sort_key(SortKey::BySongName);
                            ui_state.set_sort_ascending(true);
                            if let Some(first_song) = new_list.first() {
                                ui.invoke_play(first_song.clone(), TriggerSource::ClickItem);
                            } else {
                                let player_guard = player_clone.lock().unwrap();
                                player_guard.clear();
                                set_raw_ui_state(&ui);
                                log::warn!("song list is empty, reset UI state");
                            }
                        }
                    })
                    .unwrap();
                }
                PlayerCommand::SortSongList(key, ascending) => {
                    let ui_weak = ui_weak.clone();
                    slint::invoke_from_event_loop(move || {
                        if let Some(ui) = ui_weak.upgrade() {
                            let ui_state = ui.global::<UIState>();
                            let mut song_list: Vec<_> = ui_state.get_song_list().iter().collect();
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
                                .find(|x| x.song_path == ui_state.get_current_song().song_path)
                                .unwrap();
                            ui_state.set_current_song(new_cur_song.clone());
                            ui_state.set_sort_key(key);
                            ui_state.set_sort_ascending(ascending);
                            ui_state.set_last_sort_key(key);
                            ui_state.set_song_list(song_list.as_slice().into());
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
                            let ui_state = ui.global::<UIState>();
                            ui_state.set_lang(lang.into());
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
                            let ui_state = ui.global::<UIState>();
                            ui_state.set_volume(v);
                        }
                    })
                    .unwrap()
                }
                PlayerCommand::SetShowSpectrum(show) => {
                    spectrum_enabled.store(show, Ordering::Relaxed);
                    let ui_weak = ui_weak.clone();
                    slint::invoke_from_event_loop(move || {
                        if let Some(ui) = ui_weak.upgrade() {
                            let ui_state = ui.global::<UIState>();
                            ui_state.set_show_spectrum(show);
                            if !show {
                                ui_state.set_spectrum(default_spectrum().as_slice().into());
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
    tx.send(PlayerCommand::SetShowSpectrum(ui.global::<UIState>().get_show_spectrum()))
        .expect("failed to send initial show_spectrum command");

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
        let player_guard = player_clone.lock().unwrap();
        if let Some(ui) = ui_weak.upgrade() {
            // 如果不在拖动进度条，则自增进度条
            let ui_state = ui.global::<UIState>();
            if !ui_state.get_dragging() {
                ui_state.set_progress(player_guard.get_pos().as_secs_f32());
            }
            if !ui_state.get_paused() {
                for (idx, item) in ui_state.get_lyrics().iter().enumerate() {
                    let delta = item.time - ui_state.get_progress();
                    if delta < 0. && delta > -0.20 {
                        if idx <= 5 {
                            ui_state.set_lyric_viewport_y(0.)
                        } else {
                            ui_state.set_lyric_viewport_y(
                                (5_f32 - idx as f32) * ui_state.get_lyric_line_height(),
                            );
                        }
                        log::debug!("lyric changed to: <{:?}>", item);
                        break;
                    }
                }
            }
            // 如果播放完毕，且之前是在播放状态，则自动播放下一首
            if player_guard.empty() && ui_state.get_user_listening() && !ui_state.get_paused() {
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
            if let Some(ui) = ui_weak.upgrade()
                && let Ok(guard) = spectrum_data.lock()
            {
                let ui_state = ui.global::<UIState>();
                let show_spectrum = ui_state.get_show_spectrum();
                if !show_spectrum {
                    return;
                }
                ui_state.set_spectrum(guard.as_slice().into());
            }
        },
    );
    spectrum_timer
}

fn save_ui_state(ui: &MainWindow) {
    let ui_state = ui.global::<UIState>();
    Config::save(Config {
        song_dir: ui_state.get_song_dir().as_str().into(),
        current_song_path: Some(ui_state.get_current_song().song_path.as_str().into()),
        progress: ui_state.get_progress(),
        play_mode: ui_state.get_play_mode(),
        sort_key: ui_state.get_sort_key(),
        sort_ascending: ui_state.get_sort_ascending(),
        lang: ui_state.get_lang().into(),
        light_ui: ui_state.get_light_ui(),
        volume: ui_state.get_volume(),
        show_spectrum: ui_state.get_show_spectrum(),
    });
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
    let (fft_tx, fft_rx) = bounded::<SpectrumChunk>(4);
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
