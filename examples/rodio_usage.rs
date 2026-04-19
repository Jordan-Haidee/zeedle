use std::{thread, time::Duration};

use rodio::Decoder;
fn main() {
    // create an output stream
    let stream_handle = rodio::DeviceSinkBuilder::open_default_sink()
        .expect("no output device available");
    // create a sink to play audio
    let sink = rodio::Player::connect_new(stream_handle.mixer());
    // open an audio file
    let file = std::fs::File::open("audios/爱情转移.flac").expect("failed to open audio file");
    // decode the audio file
    let source = Decoder::try_from(file).expect("failed to decode audio file");
    // append the audio source to the sink & auto play
    sink.append(source);
    // sleep for a while to let the audio play
    thread::sleep(Duration::from_secs(20));
    // pause the audio playback explicitly
    sink.pause();
    // sleep for a while
    thread::sleep(Duration::from_secs(20));
    // resume the audio playback explicitly
    sink.play();
    // keep the main thread alive while the audio is playing
    thread::sleep(Duration::from_secs(20));
}
