//! Compression support for checkpoint data.
//!
//! Provides compression and decompression for checkpoint data to reduce storage
//! requirements and improve I/O performance.

use std::io::{Read, Write};

use crate::error::{Code, CylonError, CylonResult};

use super::config::{CompressionAlgorithm, CompressionConfig};

/// Trait for compression implementations.
pub trait Compressor: Send + Sync {
    /// Compress the input data
    fn compress(&self, data: &[u8]) -> CylonResult<Vec<u8>>;

    /// Decompress the input data
    fn decompress(&self, data: &[u8]) -> CylonResult<Vec<u8>>;

    /// Get the algorithm identifier
    fn algorithm(&self) -> CompressionAlgorithm;

    /// Get the file extension for compressed files
    fn extension(&self) -> &'static str;
}

/// No-op compressor that passes data through unchanged.
pub struct NoCompressor;

impl Compressor for NoCompressor {
    fn compress(&self, data: &[u8]) -> CylonResult<Vec<u8>> {
        Ok(data.to_vec())
    }

    fn decompress(&self, data: &[u8]) -> CylonResult<Vec<u8>> {
        Ok(data.to_vec())
    }

    fn algorithm(&self) -> CompressionAlgorithm {
        CompressionAlgorithm::None
    }

    fn extension(&self) -> &'static str {
        ""
    }
}

/// LZ4 compressor - fast compression with moderate ratio.
///
/// LZ4 is optimized for speed and is ideal for checkpointing where
/// write/read performance is critical.
pub struct Lz4Compressor {
    /// Compression level (not used by lz4_flex, but kept for API consistency)
    #[allow(dead_code)]
    level: i32,
}

impl Lz4Compressor {
    /// Create a new LZ4 compressor with default settings
    pub fn new() -> Self {
        Self { level: 0 }
    }

    /// Create a new LZ4 compressor with specified level (ignored for lz4_flex)
    pub fn with_level(level: i32) -> Self {
        Self { level }
    }
}

impl Default for Lz4Compressor {
    fn default() -> Self {
        Self::new()
    }
}

impl Compressor for Lz4Compressor {
    fn compress(&self, data: &[u8]) -> CylonResult<Vec<u8>> {
        Ok(lz4_flex::compress_prepend_size(data))
    }

    fn decompress(&self, data: &[u8]) -> CylonResult<Vec<u8>> {
        lz4_flex::decompress_size_prepended(data).map_err(|e| {
            CylonError::new(
                Code::CompressionError,
                format!("LZ4 decompression failed: {}", e),
            )
        })
    }

    fn algorithm(&self) -> CompressionAlgorithm {
        CompressionAlgorithm::Lz4
    }

    fn extension(&self) -> &'static str {
        ".lz4"
    }
}

/// Zstandard compressor - excellent compression ratio with good speed.
///
/// Zstd provides better compression ratios than LZ4 while maintaining
/// reasonable speed. Good for larger checkpoints where storage is a concern.
pub struct ZstdCompressor {
    /// Compression level (1-22, default 3)
    level: i32,
}

impl ZstdCompressor {
    /// Create a new Zstd compressor with default level (3)
    pub fn new() -> Self {
        Self { level: 3 }
    }

    /// Create a new Zstd compressor with specified level (1-22)
    pub fn with_level(level: i32) -> Self {
        Self {
            level: level.clamp(1, 22),
        }
    }
}

impl Default for ZstdCompressor {
    fn default() -> Self {
        Self::new()
    }
}

impl Compressor for ZstdCompressor {
    fn compress(&self, data: &[u8]) -> CylonResult<Vec<u8>> {
        zstd::encode_all(data, self.level).map_err(|e| {
            CylonError::new(
                Code::CompressionError,
                format!("Zstd compression failed: {}", e),
            )
        })
    }

    fn decompress(&self, data: &[u8]) -> CylonResult<Vec<u8>> {
        zstd::decode_all(data).map_err(|e| {
            CylonError::new(
                Code::CompressionError,
                format!("Zstd decompression failed: {}", e),
            )
        })
    }

    fn algorithm(&self) -> CompressionAlgorithm {
        CompressionAlgorithm::Zstd
    }

    fn extension(&self) -> &'static str {
        ".zst"
    }
}

/// Snappy compressor - very fast with moderate compression.
///
/// Snappy is designed for speed over compression ratio. It's a good
/// choice when I/O bandwidth is the bottleneck.
pub struct SnappyCompressor;

impl SnappyCompressor {
    /// Create a new Snappy compressor
    pub fn new() -> Self {
        Self
    }
}

impl Default for SnappyCompressor {
    fn default() -> Self {
        Self::new()
    }
}

impl Compressor for SnappyCompressor {
    fn compress(&self, data: &[u8]) -> CylonResult<Vec<u8>> {
        let mut encoder = snap::write::FrameEncoder::new(Vec::new());
        encoder.write_all(data).map_err(|e| {
            CylonError::new(
                Code::CompressionError,
                format!("Snappy compression failed: {}", e),
            )
        })?;
        encoder.into_inner().map_err(|e| {
            CylonError::new(
                Code::CompressionError,
                format!("Snappy compression finalize failed: {}", e),
            )
        })
    }

    fn decompress(&self, data: &[u8]) -> CylonResult<Vec<u8>> {
        let mut decoder = snap::read::FrameDecoder::new(data);
        let mut output = Vec::new();
        decoder.read_to_end(&mut output).map_err(|e| {
            CylonError::new(
                Code::CompressionError,
                format!("Snappy decompression failed: {}", e),
            )
        })?;
        Ok(output)
    }

    fn algorithm(&self) -> CompressionAlgorithm {
        CompressionAlgorithm::Snappy
    }

    fn extension(&self) -> &'static str {
        ".snappy"
    }
}

/// Create a compressor from configuration.
pub fn create_compressor(config: &CompressionConfig) -> Box<dyn Compressor> {
    match config.algorithm {
        CompressionAlgorithm::None => Box::new(NoCompressor),
        CompressionAlgorithm::Lz4 => {
            if let Some(level) = config.level {
                Box::new(Lz4Compressor::with_level(level))
            } else {
                Box::new(Lz4Compressor::new())
            }
        }
        CompressionAlgorithm::Zstd => {
            if let Some(level) = config.level {
                Box::new(ZstdCompressor::with_level(level))
            } else {
                Box::new(ZstdCompressor::new())
            }
        }
        CompressionAlgorithm::Snappy => Box::new(SnappyCompressor::new()),
    }
}

/// Create a compressor from algorithm alone (using default settings).
pub fn create_compressor_for_algorithm(algorithm: CompressionAlgorithm) -> Box<dyn Compressor> {
    match algorithm {
        CompressionAlgorithm::None => Box::new(NoCompressor),
        CompressionAlgorithm::Lz4 => Box::new(Lz4Compressor::new()),
        CompressionAlgorithm::Zstd => Box::new(ZstdCompressor::new()),
        CompressionAlgorithm::Snappy => Box::new(SnappyCompressor::new()),
    }
}

/// Detect compression algorithm from file extension.
pub fn detect_algorithm_from_extension(filename: &str) -> CompressionAlgorithm {
    if filename.ends_with(".lz4") {
        CompressionAlgorithm::Lz4
    } else if filename.ends_with(".zst") || filename.ends_with(".zstd") {
        CompressionAlgorithm::Zstd
    } else if filename.ends_with(".snappy") || filename.ends_with(".snap") {
        CompressionAlgorithm::Snappy
    } else {
        CompressionAlgorithm::None
    }
}

/// Strip compression extension from filename.
pub fn strip_compression_extension(filename: &str) -> &str {
    for ext in &[".lz4", ".zst", ".zstd", ".snappy", ".snap"] {
        if let Some(stripped) = filename.strip_suffix(ext) {
            return stripped;
        }
    }
    filename
}
