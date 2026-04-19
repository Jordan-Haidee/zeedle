use std::{
    f32::consts::PI,
    num::NonZero,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Duration,
};

use crossbeam_channel::Sender;
use rodio::Source;
use rustfft::{FftPlanner, num_complex::Complex};

pub const FFT_SIZE: usize = 256;
pub const SPECTRUM_BINS_PER_CH: usize = 24;
pub const SPECTRUM_BINS: usize = SPECTRUM_BINS_PER_CH * 2;
pub const SPECTRUM_UPDATE_MS: u64 = 100;

pub fn default_spectrum() -> Vec<f32> {
    vec![0.0; SPECTRUM_BINS]
}

pub struct SpectrumChunk {
    pub left: Vec<f32>,
    pub right: Vec<f32>,
}

pub struct TapSource<S> {
    inner: S,
    ch_count: u16,
    frame_index: u16,
    left_buf: Vec<f32>,
    right_buf: Vec<f32>,
    tx: Sender<SpectrumChunk>,
    spectrum_enabled: Arc<AtomicBool>,
}

impl<S> TapSource<S> {
    pub fn new(
        inner: S,
        tx: Sender<SpectrumChunk>,
        spectrum_enabled: Arc<AtomicBool>,
    ) -> Self
    where
        S: Source<Item = f32>,
    {
        let ch_count = inner.channels().get();
        Self {
            inner,
            ch_count,
            frame_index: 0,
            left_buf: Vec::with_capacity(FFT_SIZE),
            right_buf: Vec::with_capacity(FFT_SIZE),
            tx,
            spectrum_enabled,
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
        let channel = self.frame_index;
        self.frame_index += 1;

        if self.ch_count <= 1 {
            self.left_buf.push(sample);
            self.right_buf.push(sample);
        } else if channel == 0 {
            self.left_buf.push(sample);
        } else if channel == 1 {
            self.right_buf.push(sample);
        }

        if self.frame_index >= self.ch_count {
            self.frame_index = 0;
        }

        if !self.spectrum_enabled.load(Ordering::Relaxed) {
            self.left_buf.clear();
            self.right_buf.clear();
            return Some(sample);
        }

        if self.left_buf.len() >= FFT_SIZE && self.right_buf.len() >= FFT_SIZE {
            let mut left = Vec::with_capacity(FFT_SIZE);
            let mut right = Vec::with_capacity(FFT_SIZE);
            std::mem::swap(&mut left, &mut self.left_buf);
            std::mem::swap(&mut right, &mut self.right_buf);
            let _ = self.tx.try_send(SpectrumChunk { left, right });
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

    fn channels(&self) -> NonZero<u16> {
        self.inner.channels()
    }

    fn sample_rate(&self) -> NonZero<u32> {
        self.inner.sample_rate()
    }

    fn total_duration(&self) -> Option<Duration> {
        self.inner.total_duration()
    }

    fn try_seek(&mut self, pos: Duration) -> Result<(), rodio::source::SeekError> {
        self.frame_index = 0;
        self.left_buf.clear();
        self.right_buf.clear();
        self.inner.try_seek(pos)
    }
}

pub fn start_spectrum_worker(
    rx: crossbeam_channel::Receiver<SpectrumChunk>,
    spectrum: Arc<Mutex<Vec<f32>>>,
    spectrum_enabled: Arc<AtomicBool>,
) {
    thread::spawn(move || {
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FFT_SIZE);
        let mut buffer = vec![Complex { re: 0.0, im: 0.0 }; FFT_SIZE];
        let mut window = vec![0.0; FFT_SIZE];
        for (i, val) in window.iter_mut().enumerate() {
            *val = 0.5 - 0.5 * (2.0 * PI * i as f32 / (FFT_SIZE as f32)).cos();
        }
        let mut peak = 1e-6f32;
        let mut mags_left = vec![0.0; FFT_SIZE / 2];
        let mut mags_right = vec![0.0; FFT_SIZE / 2];

        while let Ok(chunk) = rx.recv() {
            if !spectrum_enabled.load(Ordering::Relaxed) {
                continue;
            }

            if chunk.left.len() != FFT_SIZE || chunk.right.len() != FFT_SIZE {
                continue;
            }

            for i in 0..FFT_SIZE {
                buffer[i].re = chunk.left[i] * window[i];
                buffer[i].im = 0.0;
            }
            fft.process(&mut buffer);
            for (i, c) in buffer.iter().take(FFT_SIZE / 2).enumerate() {
                mags_left[i] = (c.re * c.re + c.im * c.im).sqrt();
            }

            for i in 0..FFT_SIZE {
                buffer[i].re = chunk.right[i] * window[i];
                buffer[i].im = 0.0;
            }
            fft.process(&mut buffer);
            for (i, c) in buffer.iter().take(FFT_SIZE / 2).enumerate() {
                mags_right[i] = (c.re * c.re + c.im * c.im).sqrt();
            }

            let mut max_mag = 0.0;
            for m in mags_left.iter().chain(mags_right.iter()) {
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

            let bin_size = (mags_left.len() / SPECTRUM_BINS_PER_CH).max(1);
            let mut bars_left = vec![0.0; SPECTRUM_BINS_PER_CH];
            let mut bars_right = vec![0.0; SPECTRUM_BINS_PER_CH];
            for (b, bar) in bars_left.iter_mut().enumerate() {
                let start = b * bin_size;
                let end = if b == SPECTRUM_BINS_PER_CH - 1 {
                    mags_left.len()
                } else {
                    (b + 1) * bin_size
                };
                let mut sum = 0.0;
                for mag in mags_left.iter().take(end).skip(start) {
                    sum += mag;
                }
                let avg = sum / (end - start) as f32;
                let norm = (avg / peak).sqrt();
                *bar = norm.clamp(0.0, 1.0);
            }

            for (b, bar) in bars_right.iter_mut().enumerate() {
                let start = b * bin_size;
                let end = if b == SPECTRUM_BINS_PER_CH - 1 {
                    mags_right.len()
                } else {
                    (b + 1) * bin_size
                };
                let mut sum = 0.0;
                for mag in mags_right.iter().take(end).skip(start) {
                    sum += mag;
                }
                let avg = sum / (end - start) as f32;
                let norm = (avg / peak).sqrt();
                *bar = norm.clamp(0.0, 1.0);
            }

            peak *= 0.98;

            let mut bars = Vec::with_capacity(SPECTRUM_BINS);
            bars.extend(bars_left.into_iter().rev());
            bars.extend(bars_right);
            if let Ok(mut guard) = spectrum.lock() {
                *guard = bars;
            }
        }
    });
}
