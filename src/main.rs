#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
use std::{
    cmp::Reverse,
    f32::consts::PI,
    path::PathBuf,
    sync::{Arc, Mutex, mpsc},
    thread,
    time::{Duration, Instant},
};

use crossbeam_channel::{Sender, bounded};
use rand::Rng;
use rayon::slice::ParallelSliceMut;
use rodio::{Decoder, Source, cpal};
use rustfft::{FftPlanner, num_complex::Complex};
use slint::{Model, ToSharedString};
mod slint_types;
use slint_types::*;
mod config;
use config::Config;
mod logger;
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
}

const FFT_SIZE: usize = 1024;
const SPECTRUM_BINS: usize = 48;
const SPECTRUM_UPDATE_MS: u64 = 33;

fn default_spectrum() -> Vec<f32> {
    vec![0.0; SPECTRUM_BINS]
}

struct TapSource<S> {
    inner: S,
    channels: u16,
    frame_acc: f32,
    frame_index: u16,
    buf: Vec<f32>,
    tx: Sender<Vec<f32>>,
}

impl<S> TapSource<S> {
    fn new(inner: S, tx: Sender<Vec<f32>>) -> Self
    where
        S: Source<Item = f32>,
    {
        let channels = inner.channels();
        Self {
            inner,
            channels,
            frame_acc: 0.0,
            frame_index: 0,
            buf: Vec::with_capacity(FFT_SIZE),
            tx,
        }
    }
}

impl<S> Iterator for TapSource<S>
where
    S: Source<Item = f32>,
{
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        let sample = self.inner.next()?;
        self.frame_acc += sample;
        self.frame_index += 1;

        if self.frame_index >= self.channels {
            let mono = self.frame_acc / self.channels as f32;
            self.frame_acc = 0.0;
            self.frame_index = 0;
            self.buf.push(mono);
            if self.buf.len() >= FFT_SIZE {
                let mut chunk = Vec::with_capacity(FFT_SIZE);
                std::mem::swap(&mut chunk, &mut self.buf);
                let _ = self.tx.try_send(chunk);
            }
        }

        Some(sample)
    }
}

impl<S> Source for TapSource<S>
where
    S: Source<Item = f32>,
{
    fn current_span_len(&self) -> Option<usize> {
        self.inner.current_span_len()
    }

    fn channels(&self) -> u16 {
        self.inner.channels()
    }

    fn sample_rate(&self) -> u32 {
        self.inner.sample_rate()
    }

    fn total_duration(&self) -> Option<Duration> {
        self.inner.total_duration()
    }

    fn try_seek(&mut self, pos: Duration) -> Result<(), rodio::source::SeekError> {
        self.frame_acc = 0.0;
        self.frame_index = 0;
        self.buf.clear();
        self.inner.try_seek(pos)
    }
}

fn start_spectrum_worker(
    rx: crossbeam_channel::Receiver<Vec<f32>>,
    spectrum: Arc<Mutex<Vec<f32>>>,
) {
    thread::spawn(move || {
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FFT_SIZE);
        let mut buffer = vec![
            Complex {
                re: 0.0,
                im: 0.0
            };
            FFT_SIZE
        ];
        let mut window = vec![0.0; FFT_SIZE];
        for (i, val) in window.iter_mut().enumerate() {
            *val = 0.5 - 0.5 * (2.0 * PI * i as f32 / (FFT_SIZE as f32)).cos();
        }
        let mut peak = 1e-6f32;

        while let Ok(chunk) = rx.recv() {
            if chunk.len() != FFT_SIZE {
                continue;
            }

            for i in 0..FFT_SIZE {
                buffer[i].re = chunk[i] * window[i];
                buffer[i].im = 0.0;
            }

            fft.process(&mut buffer);

            let mut mags = vec![0.0; FFT_SIZE / 2];
            for (i, c) in buffer.iter().take(FFT_SIZE / 2).enumerate() {
                mags[i] = (c.re * c.re + c.im * c.im).sqrt();
            }

            let mut max_mag = 0.0;
            for m in &mags {
                if *m > max_mag {
                    max_mag = *m;
                }
            }
            if max_mag > peak {
                peak = max_mag;
            }
            if peak < 1e-6 {
                peak = 1e-6;
            }

            let bin_size = (mags.len() / SPECTRUM_BINS).max(1);
            let mut bars = vec![0.0; SPECTRUM_BINS];
            for (b, bar) in bars.iter_mut().enumerate() {
                let start = b * bin_size;
                let end = if b == SPECTRUM_BINS - 1 {
                    mags.len()
                } else {
                    (b + 1) * bin_size
                };
                let mut sum = 0.0;
                for mag in mags.iter().take(end).skip(start) {
                    sum += mag;
                }
                let avg = sum / (end - start) as f32;
                let norm = (avg / peak).sqrt();
                *bar = norm.clamp(0.0, 1.0);
            }

            peak *= 0.98;

            if let Ok(mut guard) = spectrum.lock() {
                *guard = bars;
            }
        }
    });
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
}

/// Set UI state according to saved config
fn set_start_ui_state(ui: &MainWindow, sink: &rodio::Sink, fft_tx: &Sender<Vec<f32>>) {
    let ui_state = ui.global::<UIState>();
    let cfg = Config::load();
    let song_list = utils::read_song_list(&cfg.song_dir, cfg.sort_key, cfg.sort_ascending);
    if song_list.is_empty() {
        log::warn!(
            "song list is empty in directory: {:?}, using default UI state ...",
            cfg.song_dir
        );
        set_raw_ui_state(ui);
        return;
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
    slint::select_bundled_translation(&cfg.lang)
        .unwrap_or_else(|_| panic!("failed to set language: {}", cfg.lang));
    ui_state.set_song_list(song_list.as_slice().into());
    ui_state.set_song_dir(cfg.song_dir.to_str().expect("failed to convert Path to String").into());
    ui_state.set_about_info(utils::get_about_info());
    let mut cur_song_info = utils::read_meta_info(
        cfg.current_song_path.unwrap_or(song_list[0].song_path.as_str().into()),
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
    let file = std::fs::File::open(&cur_song_info.song_path)
        .unwrap_or_else(|_| panic!("failed to open audio file: {}", cur_song_info.song_path));
    let source = Decoder::try_from(file).expect("failed to decode audio file");
    let tap = TapSource::new(source, fft_tx.clone());
    sink.append(tap);
    sink.pause();
    sink.set_volume(cfg.volume);
    sink.try_seek(Duration::from_secs_f32(cfg.progress)).expect("failed to seek to given position");
    let mut history = ui_state.get_play_history().iter().collect::<Vec<_>>();
    history.push(cur_song_info.clone());
    ui_state.set_play_history(history.as_slice().into());
    ui_state.set_history_index(0);
}

fn main() {
    let app_start = Instant::now();
    logger::init_default_logger(None::<PathBuf>);
    // when panics happen, auto port errors to log
    std::panic::set_hook(Box::new(|info| {
        log::error!("{}", info);
    }));
    let ins = single_instance::SingleInstance::new("Zeedle Music Player").unwrap();
    if !ins.is_single() {
        log::warn!("Vanilla player can only run one instance !");
        return;
    }
    let mut stream_handle = rodio::OutputStreamBuilder::from_default_device()
        .expect("no output device available")
        .with_buffer_size(cpal::BufferSize::Fixed(4096))
        .open_stream()
        .expect("failed to open output stream");
    stream_handle.log_on_drop(false);
    let _sink = rodio::Sink::connect_new(stream_handle.mixer());
    let sink = Arc::new(Mutex::new(_sink));
    let spectrum_data = Arc::new(Mutex::new(default_spectrum()));
    let (fft_tx, fft_rx) = bounded::<Vec<f32>>(4);
    start_spectrum_worker(fft_rx, spectrum_data.clone());
    // 创建消息通道 ui --> backend
    let (tx, rx) = mpsc::channel::<PlayerCommand>();
    // 初始化 UI 状态
    let ui = MainWindow::new().expect("failed to create UI");
    set_start_ui_state(&ui, &sink.lock().unwrap(), &fft_tx);

    // 播放线程
    let ui_weak = ui.as_weak();
    let sink_clone = sink.clone();
    let fft_tx = fft_tx.clone();
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
                    let sink_guard = sink_clone.lock().unwrap();
                    sink_guard.clear();
                    let tap = TapSource::new(source, fft_tx.clone());
                    sink_guard.append(tap);
                    sink_guard.play();
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
                    let sink_guard = sink_clone.lock().unwrap();
                    let ui_weak = ui_weak.clone();
                    if sink_guard.empty() {
                        log::info!("sink is empty, play the first song in the list");
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
                        let paused = sink_guard.is_paused();
                        if paused {
                            sink_guard.play();
                        } else {
                            sink_guard.pause();
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
                    let sink_guard = sink_clone.lock().unwrap();
                    match sink_guard.try_seek(Duration::from_secs_f32(new_progress)) {
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
                    let sink_clone = sink_clone.clone();
                    slint::invoke_from_event_loop(move || {
                        if let Some(ui) = ui_weak.upgrade() {
                            let ui_state = ui.global::<UIState>();
                            ui_state.set_song_list(new_list.as_slice().into());
                            ui_state.set_sort_key(SortKey::BySongName);
                            ui_state.set_sort_ascending(true);
                            if let Some(first_song) = new_list.first() {
                                ui.invoke_play(first_song.clone(), TriggerSource::ClickItem);
                            } else {
                                let sink_guard = sink_clone.lock().unwrap();
                                sink_guard.clear();
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
                    let sink_guard = sink_clone.lock().unwrap();
                    sink_guard.set_volume(v);
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
            }
        }
    });

    // UI 触发事件
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
    // pure callback to format duration string
    ui.on_format_duration(|dura| {
        format!("{:02}:{:02}", (dura as u32) / 60, (dura as u32) % 60).to_shared_string()
    });
    // UI 定时刷新进度条
    let ui_weak = ui.as_weak();
    let timer = slint::Timer::default();
    let sink_clone = sink.clone();
    timer.start(slint::TimerMode::Repeated, Duration::from_millis(200), move || {
        let sink_guard = sink_clone.lock().unwrap();
        if let Some(ui) = ui_weak.upgrade() {
            // 如果不在拖动进度条，则自增进度条
            let ui_state = ui.global::<UIState>();
            if !ui_state.get_dragging() {
                ui_state.set_progress(sink_guard.get_pos().as_secs_f32());
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
            if sink_guard.empty() && ui_state.get_user_listening() && !ui_state.get_paused() {
                ui.invoke_play_next();
                log::info!("song ended, auto play next");
            }
        }
    });

    // UI 定时刷新频谱
    let ui_weak = ui.as_weak();
    let spectrum_data = spectrum_data.clone();
    let spectrum_timer = slint::Timer::default();
    spectrum_timer.start(
        slint::TimerMode::Repeated,
        Duration::from_millis(SPECTRUM_UPDATE_MS),
        move || {
            if let Some(ui) = ui_weak.upgrade()
                && let Ok(guard) = spectrum_data.lock()
            {
                let ui_state = ui.global::<UIState>();
                ui_state.set_spectrum(guard.as_slice().into());
            }
        },
    );

    // 显示 UI
    log::info!("ui state initialized, take: {:?}", app_start.elapsed());
    ui.run().expect("failed to run UI");

    // 退出前保存状态
    log::info!("saving config...");
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
    });
    log::info!("app exited");
}
