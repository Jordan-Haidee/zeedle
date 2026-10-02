use std::{fs::File, io::BufReader, path::Path};

use lofty::{
    config::WriteOptions,
    error::{ErrorKind as LoftyErrorKind, LoftyError},
    file::{AudioFile, TaggedFileExt},
    picture::{MimeType, Picture, PictureType},
    probe::Probe,
    tag::{Accessor, ItemKey, Tag, TagExt},
};
use pinyin::ToPinyin;
use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
    },
    slice::ParallelSliceMut,
};
use slint::{StyledText, ToSharedString};

use crate::slint_types::{LyricItem, SongInfo, SortKey};

#[derive(Debug)]
pub struct CoverPreview {
    pub rgba: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

pub struct UploadedCover {
    pub picture: Picture,
    pub preview: CoverPreview,
}

pub struct SongEditMetadata {
    pub album: String,
    pub lyrics: String,
    pub cover: Option<CoverPreview>,
}

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

/// Read the fields shown by the song editor, including the existing embedded cover and lyrics.
pub fn read_song_edit_metadata(path: impl AsRef<Path>) -> Result<SongEditMetadata, LoftyError> {
    let tagged_file = lofty::read_from_path(path.as_ref())?;
    let tag = tagged_file.primary_tag();
    let album = tag.and_then(Accessor::album).map(|value| value.into_owned()).unwrap_or_default();
    let lyrics = tag
        .and_then(|tag| tag.get(ItemKey::UnsyncLyrics).or_else(|| tag.get(ItemKey::Lyrics)))
        .and_then(|item| item.value().text())
        .unwrap_or_default()
        .to_owned();
    let cover = tag
        .and_then(|tag| {
            tag.get_picture_type(PictureType::CoverFront)
                .or_else(|| tag.get_picture_type(PictureType::CoverBack))
        })
        .and_then(|picture| image::load_from_memory(picture.data()).ok())
        .map(|image| {
            let rgba = image.into_rgba8();
            let (width, height) = rgba.dimensions();
            CoverPreview {
                rgba: rgba.into_vec(),
                width,
                height,
            }
        });

    Ok(SongEditMetadata {
        album,
        lyrics,
        cover,
    })
}

/// Validate a PNG or JPEG upload and prepare both its embedded tag and UI preview.
pub fn load_uploaded_cover(path: impl AsRef<Path>) -> Result<UploadedCover, LoftyError> {
    let path = path.as_ref();
    let extension = path.extension().and_then(|ext| ext.to_str()).unwrap_or_default();
    let (expected_format, mime_type) = match extension.to_ascii_lowercase().as_str() {
        "png" => (image::ImageFormat::Png, MimeType::Png),
        "jpg" | "jpeg" => (image::ImageFormat::Jpeg, MimeType::Jpeg),
        _ => return Err(LoftyError::new(LoftyErrorKind::NotAPicture)),
    };

    let bytes = std::fs::read(path)?;
    if image::guess_format(&bytes).ok() != Some(expected_format) {
        return Err(LoftyError::new(LoftyErrorKind::NotAPicture));
    }
    let decoded = image::load_from_memory(&bytes)
        .map_err(|_| LoftyError::new(LoftyErrorKind::NotAPicture))?;
    let rgba = decoded.into_rgba8();
    let (width, height) = rgba.dimensions();
    let picture =
        Picture::unchecked(bytes).pic_type(PictureType::CoverFront).mime_type(mime_type).build();

    Ok(UploadedCover {
        picture,
        preview: CoverPreview {
            rgba: rgba.into_vec(),
            width,
            height,
        },
    })
}

/// Read UTF-8 LRC contents and reject empty or wrongly suffixed uploads.
pub fn read_uploaded_lyrics(path: impl AsRef<Path>) -> Result<String, LoftyError> {
    let path = path.as_ref();
    if !path
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("lrc"))
    {
        return Err(LoftyError::new(LoftyErrorKind::TextDecode(
            "lyrics file must use the .lrc extension",
        )));
    }

    let bytes = std::fs::read(path)?;
    let text = String::from_utf8(bytes)?;
    let lyrics = text.strip_prefix('\u{feff}').unwrap_or(&text);
    if lyrics.trim().is_empty() {
        return Err(LoftyError::new(LoftyErrorKind::TextDecode("LRC file is empty")));
    }
    Ok(lyrics.to_owned())
}

/// Write the title, artist, album, and any selected cover or lyrics uploads to the audio tag.
pub fn write_song_metadata(
    path: impl AsRef<Path>,
    title: &str,
    artist: &str,
    album: &str,
    cover_path: Option<&Path>,
    lyrics_path: Option<&Path>,
) -> Result<(), LoftyError> {
    let path = path.as_ref();
    // Validate selected files before changing any tag fields.
    let cover = cover_path.map(load_uploaded_cover).transpose()?;
    let lyrics = lyrics_path.map(read_uploaded_lyrics).transpose()?;
    let mut tagged_file = lofty::read_from_path(path)?;
    let tag_type = tagged_file.primary_tag_type();
    if tagged_file.primary_tag_mut().is_none() {
        tagged_file.insert_tag(Tag::new(tag_type));
    }

    let tag = tagged_file.primary_tag_mut().expect("primary tag was inserted");
    tag.set_title(title.to_owned());
    tag.set_artist(artist.to_owned());
    if album.trim().is_empty() {
        tag.remove_album();
    } else {
        tag.set_album(album.trim().to_owned());
    }
    if let Some(cover) = cover {
        tag.remove_picture_type(PictureType::CoverFront);
        tag.push_picture(cover.picture);
    }
    if let Some(lyrics) = lyrics
        && !tag.insert_text(ItemKey::UnsyncLyrics, lyrics)
    {
        return Err(LoftyError::new(LoftyErrorKind::UnsupportedTag));
    }
    tag.save_to_path(path, WriteOptions::default())
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

/// Sort a song list using the selected key and refresh each row's index.
pub fn sort_song_infos(songs: &mut [SongInfo], sort_key: SortKey, ascending: bool) {
    match sort_key {
        SortKey::BySongName => {
            let key_fn = |song: &SongInfo| get_chars(song.song_name.as_str());
            if ascending {
                songs.par_sort_by_key(key_fn);
            } else {
                songs.par_sort_by_key(|song| std::cmp::Reverse(key_fn(song)));
            }
        }
        SortKey::BySinger => {
            let key_fn = |song: &SongInfo| get_chars(song.singer.as_str());
            if ascending {
                songs.par_sort_by_key(key_fn);
            } else {
                songs.par_sort_by_key(|song| std::cmp::Reverse(key_fn(song)));
            }
        }
        SortKey::ByDuration => {
            if ascending {
                songs.par_sort_by_key(|song| song.duration.clone());
            } else {
                songs.par_sort_by_key(|song| std::cmp::Reverse(song.duration.clone()));
            }
        }
    }
    songs.iter_mut().enumerate().for_each(|(index, song)| song.id = index as i32);
}

/// Read lyrics from audio file `p`, return a list of LyricItem
pub fn read_lyrics(path: impl AsRef<Path>) -> Vec<LyricItem> {
    let path = path.as_ref();

    let probe = match open_audio_probe(path) {
        Ok(p) => p,
        Err(e) => {
            log::warn!("{}: failed to open or detect file type ({})", path.display(), e);
            return vec![];
        }
    };
    if let Ok(tagged) = probe.read()
        && let Some(tag) = tagged.primary_tag()
        && let Some(lyric_item) =
            tag.get(ItemKey::UnsyncLyrics).or_else(|| tag.get(ItemKey::Lyrics))
        && let Some(lyrics) = lyric_item.value().text()
    {
        return parse_lyrics(lyrics);
    }
    Vec::new()
}

/// Parse time-coded LRC text into the timeline items used by the lyrics panel.
pub fn parse_lyrics(lyrics: &str) -> Vec<LyricItem> {
    let mut lyrics = lyrics
        .trim_start_matches('\u{feff}')
        .lines()
        .filter_map(|line| {
            let (time_str, text) = line.split_once(']')?;
            let time_str = time_str.trim_start_matches('[');
            let duration = time_str
                .split(':')
                .map(|part| part.parse::<f32>().ok())
                .collect::<Option<Vec<_>>>()?
                .into_iter()
                .rev()
                .reduce(|acc, value| acc + value * 60.0)?;
            if duration <= 0.0 || text.is_empty() {
                return None;
            }
            Some(LyricItem {
                time: duration,
                text: text.trim_end().to_shared_string(),
                duration: 0.0,
            })
        })
        .collect::<Vec<_>>();

    for index in 0..lyrics.len().saturating_sub(1) {
        lyrics[index].duration = lyrics[index + 1].time - lyrics[index].time;
    }
    if let Some(last) = lyrics.last_mut() {
        last.duration = 100.0;
    }
    lyrics
}

/// Read album cover from audio file `p`, return a slint::Image
pub fn read_album_cover(path: impl AsRef<Path>) -> Option<(Vec<u8>, u32, u32)> {
    let path = path.as_ref();
    if let Ok(probe) = open_audio_probe(path)
        && let Ok(tagged) = probe.read()
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
pub fn get_about_info() -> StyledText {
    let name = env!("CARGO_PKG_NAME");
    let s = format!(
        "{}\n{}\nAuthor: {}\nVersion: {}\n[Github]({})",
        name.get(..1).map(|c| c.to_uppercase() + &name[1..]).unwrap_or_default(),
        env!("CARGO_PKG_DESCRIPTION"),
        "Jordan Haidee",
        env!("CARGO_PKG_VERSION"),
        env!("CARGO_PKG_REPOSITORY"),
    );
    slint::StyledText::from_markdown(&s).unwrap()
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
        .par_iter()
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

#[cfg(test)]
mod metadata_upload_tests {
    use std::{
        io::Cursor,
        path::{Path, PathBuf},
        time::{SystemTime, UNIX_EPOCH},
    };

    use lofty::{
        file::TaggedFileExt,
        picture::{MimeType, PictureType},
        tag::{Accessor, ItemKey},
    };

    use super::{read_song_edit_metadata, write_song_metadata};

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            let unique = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
            let path = std::env::temp_dir()
                .join(format!("zeedle-metadata-{}-{unique}", std::process::id()));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn path(&self, name: &str) -> PathBuf {
            self.0.join(name)
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn write_minimal_wav(path: &Path) {
        let sample_count = 8_000u32;
        let data_size = sample_count * 2;
        let mut wav = Vec::with_capacity(44 + data_size as usize);
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data_size).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
        wav.extend_from_slice(&1u16.to_le_bytes()); // mono
        wav.extend_from_slice(&8_000u32.to_le_bytes());
        wav.extend_from_slice(&16_000u32.to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_size.to_le_bytes());
        wav.resize(44 + data_size as usize, 0);
        std::fs::write(path, wav).unwrap();
    }

    fn write_test_jpeg(path: &Path) -> Vec<u8> {
        let image = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            4,
            4,
            image::Rgb([20, 40, 60]),
        ));
        let mut encoded = Cursor::new(Vec::new());
        image.write_to(&mut encoded, image::ImageFormat::Jpeg).unwrap();
        let bytes = encoded.into_inner();
        std::fs::write(path, &bytes).unwrap();
        bytes
    }

    fn write_lrc(path: &Path, contents: &str) {
        std::fs::write(path, contents).unwrap();
    }

    #[test]
    fn metadata_uploads_roundtrip_album_png_jpg_and_lrc() {
        let temp = TestDirectory::new();
        let audio = temp.path("track.wav");
        let png = temp.path("cover.png");
        let jpg = temp.path("cover.jpg");
        let first_lrc = temp.path("first.lrc");
        let second_lrc = temp.path("second.lrc");
        write_minimal_wav(&audio);
        let png_bytes = include_bytes!("../ui/cover.png");
        std::fs::write(&png, png_bytes).unwrap();
        write_lrc(&first_lrc, "[00:01.00]First line\n[00:02.00]Second line\n");

        write_song_metadata(
            &audio,
            "First title",
            "First artist",
            "First album",
            Some(&png),
            Some(&first_lrc),
        )
        .unwrap();

        let first = lofty::read_from_path(&audio).unwrap();
        let first_tag = first.primary_tag().unwrap();
        assert_eq!(first_tag.title().as_deref(), Some("First title"));
        assert_eq!(first_tag.artist().as_deref(), Some("First artist"));
        assert_eq!(first_tag.album().as_deref(), Some("First album"));
        assert_eq!(
            first_tag.get(ItemKey::UnsyncLyrics).and_then(|item| item.value().text()),
            Some("[00:01.00]First line\n[00:02.00]Second line\n")
        );
        let first_cover = first_tag.get_picture_type(PictureType::CoverFront).unwrap();
        assert_eq!(first_cover.mime_type(), Some(&MimeType::Png));
        assert_eq!(first_cover.data(), png_bytes);

        let first_edit_info = read_song_edit_metadata(&audio).unwrap();
        assert_eq!(first_edit_info.album, "First album");
        assert!(first_edit_info.cover.is_some());
        assert_eq!(first_edit_info.lyrics.lines().count(), 2);

        let jpg_bytes = write_test_jpeg(&jpg);
        write_lrc(&second_lrc, "[00:03.00]Replacement line\n");
        write_song_metadata(
            &audio,
            "Updated title",
            "Updated artist",
            "Updated album",
            Some(&jpg),
            Some(&second_lrc),
        )
        .unwrap();

        let second = lofty::read_from_path(&audio).unwrap();
        let second_tag = second.primary_tag().unwrap();
        assert_eq!(second_tag.title().as_deref(), Some("Updated title"));
        assert_eq!(second_tag.artist().as_deref(), Some("Updated artist"));
        assert_eq!(second_tag.album().as_deref(), Some("Updated album"));
        assert_eq!(
            second_tag.get(ItemKey::UnsyncLyrics).and_then(|item| item.value().text()),
            Some("[00:03.00]Replacement line\n")
        );
        let second_cover = second_tag.get_picture_type(PictureType::CoverFront).unwrap();
        assert_eq!(second_cover.mime_type(), Some(&MimeType::Jpeg));
        assert_eq!(second_cover.data(), jpg_bytes);

        write_song_metadata(&audio, "Third title", "Third artist", "Third album", None, None)
            .unwrap();
        let unchanged_assets = lofty::read_from_path(&audio).unwrap();
        let tag = unchanged_assets.primary_tag().unwrap();
        assert_eq!(tag.title().as_deref(), Some("Third title"));
        assert_eq!(tag.album().as_deref(), Some("Third album"));
        assert_eq!(
            tag.get(ItemKey::UnsyncLyrics).and_then(|item| item.value().text()),
            Some("[00:03.00]Replacement line\n")
        );
        assert_eq!(tag.get_picture_type(PictureType::CoverFront).unwrap().data(), jpg_bytes);
    }

    #[test]
    fn invalid_uploads_do_not_partially_modify_audio_tags() {
        let temp = TestDirectory::new();
        let audio = temp.path("track.wav");
        let invalid_cover = temp.path("broken.png");
        let invalid_lrc = temp.path("broken.lrc");
        write_minimal_wav(&audio);
        write_song_metadata(&audio, "Original", "Artist", "Album", None, None).unwrap();
        std::fs::write(&invalid_cover, b"not an image").unwrap();
        std::fs::write(&invalid_lrc, [0xFF, 0xFE]).unwrap();

        assert!(
            write_song_metadata(
                &audio,
                "Must not be written",
                "Artist",
                "Album",
                Some(&invalid_cover),
                None,
            )
            .is_err()
        );
        assert!(
            write_song_metadata(
                &audio,
                "Must not be written",
                "Artist",
                "Album",
                None,
                Some(&invalid_lrc),
            )
            .is_err()
        );

        let tagged = lofty::read_from_path(&audio).unwrap();
        let tag = tagged.primary_tag().unwrap();
        assert_eq!(tag.title().as_deref(), Some("Original"));
        assert_eq!(tag.artist().as_deref(), Some("Artist"));
        assert_eq!(tag.album().as_deref(), Some("Album"));
    }
}
