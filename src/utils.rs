use std::path::Path;

use globset::GlobBuilder;
use lofty::{
    file::{AudioFile, TaggedFileExt},
    picture::PictureType,
    tag::{Accessor, ItemKey},
};
use pinyin::ToPinyin;
use rayon::{
    iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
use slint::{SharedString, ToSharedString};
use walkdir::WalkDir;

use crate::slint_types::{LyricItem, SongInfo, SortKey};

/// Read meta info from audio file `fp`, return a SongInfo
pub fn read_meta_info(path: impl AsRef<Path>) -> Option<SongInfo> {
    let path = path.as_ref();
    if let Ok(tagged) = lofty::read_from_path(path) {
        let dura = tagged.properties().duration().as_secs_f32();
        if let Some(tag) = tagged.primary_tag() {
            let song_name = tag.title();
            let song_name = song_name
                .as_deref()
                .unwrap_or(path.file_stem().and_then(|x| x.to_str()).unwrap_or("unknown"));
            let singer_name = tag.artist();
            let singer_name = singer_name.as_deref().unwrap_or("unknown");

            let item = SongInfo {
                id: 0,
                song_path: path.display().to_shared_string(),
                song_name: song_name.into(),
                singer: singer_name.into(),
                duration: format!("{:02}:{:02}", (dura as u32) / 60, (dura as u32) % 60)
                    .to_shared_string(),
            };
            return Some(item);
        }
    }
    None
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
    let glober = GlobBuilder::new("**/*.{mp3,flac,wav,ogg}").build().unwrap().compile_matcher();
    let entries = WalkDir::new(audio_dir)
        .into_iter()
        .filter_map(|x| x.ok())
        .filter(|x| glober.is_match(x.path()))
        .collect::<Vec<_>>();
    let mut songs = entries
        .into_par_iter()
        .map(|entry| read_meta_info(entry.path()))
        .flatten()
        .collect::<Vec<_>>();
    if ascending {
        match sort_key {
            SortKey::BySongName => songs.par_sort_by_key(|x| get_chars(x.song_name.as_str())),
            SortKey::BySinger => songs.par_sort_by_key(|x| get_chars(x.singer.as_str())),
            SortKey::ByDuration => songs.par_sort_by_key(|x| x.duration.clone()),
        }
    } else {
        match sort_key {
            SortKey::BySongName => {
                songs.par_sort_by_key(|x| std::cmp::Reverse(get_chars(x.song_name.as_str())))
            }
            SortKey::BySinger => {
                songs.par_sort_by_key(|x| std::cmp::Reverse(get_chars(x.singer.as_str())))
            }
            SortKey::ByDuration => songs.par_sort_by_key(|x| std::cmp::Reverse(x.duration.clone())),
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
    if let Ok(tagged) = lofty::read_from_path(path)
        && let Some(tag) = tagged.primary_tag()
        && let Some(lyric_item) = tag.get(&ItemKey::Lyrics)
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

pub fn from_image_to_slint(buffer: Vec<u8>, width: u32, height: u32) -> slint::Image {
    let mut pixel_buffer = slint::SharedPixelBuffer::new(width, height);
    let pixel_buffer_data = pixel_buffer.make_mut_bytes();
    pixel_buffer_data.copy_from_slice(&buffer);
    slint::Image::from_rgba8(pixel_buffer)
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
