use std::{fs::File, io::BufReader, path::Path};

use lofty::{
    error::LoftyError,
    file::{AudioFile, TaggedFileExt},
    picture::PictureType,
    probe::Probe,
    tag::{Accessor, ItemKey},
};
use pinyin::ToPinyin;
use rayon::{
    iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
use slint::{SharedString, ToSharedString};

use crate::slint_types::{LyricItem, SongInfo, SortKey};

/// Open an audio file and detect its type, returning a configured [`Probe`] ready to read.
fn open_audio_probe(path: &Path) -> Result<Probe<BufReader<File>>, LoftyError> {
    Probe::open(path)?.guess_file_type().map_err(|e| e.into())
}

/// Read meta info from audio file `fp`, return a SongInfo
pub fn read_meta_info(path: impl AsRef<Path>) -> Result<SongInfo, LoftyError> {
    let path = path.as_ref();

    let probe = open_audio_probe(path)?;
    match probe.read() {
        Ok(tagged) => {
            let file_stem = path.file_stem().and_then(|x| x.to_str()).unwrap_or("unknown");
            let dura = tagged.properties().duration().as_secs_f32();
            let dura_str =
                format!("{:02}:{:02}", (dura as u32) / 60, (dura as u32) % 60).to_shared_string();
            match tagged.primary_tag() {
                Some(tag) => {
                    let song_name = tag.title();
                    let song_name = song_name.as_deref().unwrap_or(file_stem);
                    let singer_name = tag.artist();
                    let singer_name = singer_name.as_deref().unwrap_or("unknown");

                    let item = SongInfo {
                        id: 0,
                        song_path: path.display().to_shared_string(),
                        song_name: song_name.into(),
                        singer: singer_name.into(),
                        duration: dura_str,
                    };
                    Ok(item)
                }
                None => {
                    let item = SongInfo {
                        id: 0,
                        song_path: path.display().to_shared_string(),
                        song_name: file_stem.into(),
                        singer: "unknown".into(),
                        duration: dura_str,
                    };
                    Ok(item)
                }
            }
        }
        Err(e) => Err(e),
    }
}

/// Scan songs in Path `p` and return a list of SongInfo
pub fn read_song_list(
    audio_dir: impl AsRef<Path>,
    sort_key: SortKey,
    ascending: bool,
) -> Vec<SongInfo> {
    let audio_dir = audio_dir.as_ref();
    if !audio_dir.exists() {
        return Vec::new();
    }
    let entries = std::fs::read_dir(audio_dir)
        .into_iter()
        .flatten()
        .filter_map(|x| x.ok())
        .filter(|x| {
            x.path()
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| matches!(e, "mp3" | "flac" | "wav" | "ogg"))
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    let mut songs = entries
        .into_par_iter()
        .map(|entry| entry.path())
        .map(|entry| (read_meta_info(&entry), entry))
        .filter_map(|(r, p)| {
            r.map_or_else(
                |e| {
                    log::warn!("error when reading meta info from {} : {}", p.display(), e);
                    None
                },
                Some,
            )
        })
        .collect::<Vec<_>>();
    match sort_key {
        SortKey::BySongName => {
            let key_fn = |x: &SongInfo| get_chars(x.song_name.as_str());
            if ascending {
                songs.par_sort_by_key(key_fn);
            } else {
                songs.par_sort_by_key(|x| std::cmp::Reverse(key_fn(x)));
            }
        }
        SortKey::BySinger => {
            let key_fn = |x: &SongInfo| get_chars(x.singer.as_str());
            if ascending {
                songs.par_sort_by_key(key_fn);
            } else {
                songs.par_sort_by_key(|x| std::cmp::Reverse(key_fn(x)));
            }
        }
        SortKey::ByDuration => {
            if ascending {
                songs.par_sort_by_key(|x| x.duration.clone());
            } else {
                songs.par_sort_by_key(|x| std::cmp::Reverse(x.duration.clone()));
            }
        }
    }
    songs
        .into_par_iter()
        .enumerate()
        .map(|(idx, mut x)| {
            x.id = idx as i32;
            x
        })
        .collect::<Vec<_>>()
}

/// Read lyrics from audio file `p`, return a list of LyricItem
pub fn read_lyrics(path: impl AsRef<Path>) -> Vec<LyricItem> {
    let path = path.as_ref();

    let probe = match Probe::open(path) {
        Ok(p) => p,
        Err(e) => {
            log::warn!("{}: failed to open file ({})", path.display(), e);
            return vec![];
        }
    };
    let probe = match probe.guess_file_type() {
        Ok(p) => p,
        Err(e) => {
            log::warn!("{}: failed to detect file type ({})", path.display(), e);
            return vec![];
        }
    };
    if let Ok(tagged) = probe.read()
        && let Some(tag) = tagged.primary_tag()
        && let Some(lyric_item) =
            tag.get(ItemKey::UnsyncLyrics).or_else(|| tag.get(ItemKey::Lyrics))
    {
        let mut lyrics = lyric_item
            .value()
            .text()
            .unwrap()
            .split("\n")
            .map(|line| {
                let (time_str, text) = line.split_once(']').unwrap_or(("", ""));
                let time_str = time_str.trim_start_matches('[');
                let dura = time_str
                    .split(':')
                    .map(|x| x.parse::<f32>().unwrap_or(0.))
                    .rev()
                    .reduce(|acc, x| acc + x * 60.)
                    .unwrap_or(0.);
                LyricItem {
                    time: dura,
                    text: text.to_shared_string(),
                    duration: 0.0,
                }
            })
            .filter(|ins| ins.time > 0. && !ins.text.is_empty())
            .collect::<Vec<_>>();
        for i in 0..lyrics.len() - 1 {
            lyrics[i].duration = lyrics[i + 1].time - lyrics[i].time;
        }
        if let Some(ins) = lyrics.last_mut() {
            ins.duration = 100.0;
        }
        return lyrics;
    }
    Vec::new()
}

/// Read album cover from audio file `p`, return a slint::Image
pub fn read_album_cover(path: impl AsRef<Path>) -> Option<(Vec<u8>, u32, u32)> {
    let path = path.as_ref();
    if let Ok(tagged) = lofty::read_from_path(path)
        && let Some(tag) = tagged.primary_tag()
        && let Some(picture) = tag.pictures().iter().find(|pic| {
            pic.pic_type() == PictureType::CoverFront || pic.pic_type() == PictureType::CoverBack
        })
        && let Ok(img) = image::load_from_memory(picture.data())
    {
        let rgba = img.into_rgba8();
        let (width, height) = rgba.dimensions();
        let buffer = rgba.into_vec();
        return Some((buffer, width, height));
    }
    None
}

pub fn get_default_album_cover() -> slint::Image {
    slint::Image::load_from_svg_data(include_bytes!("../ui/cover.svg")).unwrap()
}

/// Get about info string
pub fn get_about_info() -> SharedString {
    format!(
        "{}\n{}\nAuthor: {}\nVersion: {}",
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_DESCRIPTION"),
        env!("CARGO_PKG_AUTHORS"),
        env!("CARGO_PKG_VERSION")
    )
    .into()
}

/// obtain sort key
/// 1. english -> fist char
/// 2. chinese -> first pinyin char
/// 3. other -> ~
pub fn get_chars(s: &str) -> (u8, String) {
    // obtain first char
    let first_char = s.chars().next();
    match first_char {
        Some(c) if c.is_ascii_alphabetic() => {
            // en: first group
            (0, s.to_string())
        }
        Some(c) if is_chinese(c) => {
            // cn: second group
            let pinyin = s
                .to_pinyin()
                .flatten()
                .map(|p| p.plain().to_string())
                .collect::<Vec<String>>()
                .join("");
            (1, pinyin)
        }
        _ => {
            // other: third group
            (2, "&".to_string())
        }
    }
}

/// judge whether a char is chinese
fn is_chinese(c: char) -> bool {
    ('\u{4e00}'..='\u{9fff}').contains(&c)
}

/// Search songs by query text, returns results sorted by relevance.
/// Matches against song_name and singer (case-insensitive).
pub fn search_songs(query: &str, songs: &[SongInfo]) -> Vec<SongInfo> {
    if query.is_empty() {
        return Vec::new();
    }
    let query_lower = query.to_lowercase();

    struct Scored {
        song: SongInfo,
        score: i32,
    }

    let mut scored: Vec<Scored> = songs
        .iter()
        .filter_map(|song| {
            let name_lower = song.song_name.as_str().to_lowercase();
            let singer_lower = song.singer.as_str().to_lowercase();

            let mut score = 0i32;

            // Score from song_name matching
            if name_lower == query_lower {
                score += 100;
            } else if name_lower.starts_with(&query_lower) {
                score += 80;
            } else if name_lower.contains(&query_lower) {
                score += 60;
            }

            // Score from singer matching
            if singer_lower == query_lower {
                score += 50;
            } else if singer_lower.starts_with(&query_lower) {
                score += 40;
            } else if singer_lower.contains(&query_lower) {
                score += 20;
            }

            if score > 0 {
                Some(Scored {
                    song: song.clone(),
                    score,
                })
            } else {
                None
            }
        })
        .collect();

    // Sort by score descending
    scored.par_sort_by_key(|s| std::cmp::Reverse(s.score));

    scored.into_iter().map(|s| s.song).collect()
}

/// Get default font family for different platforms to ensure proper rendering of Chinese characters.
pub fn get_default_font_family() -> &'static str {
    #[cfg(target_os = "windows")]
    return "Microsoft YaHei UI";

    #[cfg(target_os = "linux")]
    return "Noto Sans CJK SC";

    #[cfg(target_os = "macos")]
    return "PingFang SC";
}
