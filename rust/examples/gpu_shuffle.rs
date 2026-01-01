// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! GPU Shuffle Example
//!
//! Demonstrates distributed shuffle operation using gcylon.
//! Similar to the C++ gshuffle example.
//!
//! Run with: mpirun -n 2 cargo run --example gpu_shuffle --features gpu

use std::time::Instant;
use cylon::gpu::{GpuContext, GpuConfig, get_device_count, set_device};
use cylon::error::CylonResult;

fn main() -> CylonResult<()> {
    // Parse command line args
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <data_size>", args[0]);
        eprintln!("  data_size: Total data size like 100MB, 1GB");
        std::process::exit(1);
    }

    let data_size = &args[1];

    // Initialize GPU context with MPI
    let ctx = GpuContext::new_mpi()?;
    let rank = ctx.rank();
    let world_size = ctx.world_size();

    // Set GPU device based on rank
    let num_gpus = get_device_count()?;
    let device_id = rank % num_gpus;
    set_device(device_id)?;
    println!("Rank {}: Using GPU {} of {}", rank, device_id, num_gpus);

    // Calculate rows based on data size
    let cols = 4;
    let rows = calculate_rows(data_size, cols, world_size)?;

    println!("Rank {}: Creating table with {} columns, {} rows", rank, cols, rows);

    // Create sequential table
    let table = ctx.create_sequential_table(cols, rows, 0, 1)?;
    println!(
        "Rank {}: Created table with {} rows, {} columns",
        rank,
        table.num_rows(),
        table.num_columns()
    );

    // Perform shuffle on column 0
    let config = GpuConfig::default();
    let start = Instant::now();
    let shuffled = table.shuffle(&[0], Some(config))?;
    let duration = start.elapsed();

    println!(
        "Rank {}: Shuffle completed in {:?}, output has {} rows",
        rank,
        duration,
        shuffled.num_rows()
    );

    // Print memory info
    if let Ok(mem) = ctx.memory_info() {
        println!(
            "Rank {}: GPU memory - used: {} MB, free: {} MB, total: {} MB",
            rank,
            mem.used / (1024 * 1024),
            mem.free / (1024 * 1024),
            mem.total / (1024 * 1024)
        );
    }

    if rank == 0 {
        println!("\n========================================");
        println!("GPU SHUFFLE COMPLETED SUCCESSFULLY");
        println!("========================================\n");
    }

    Ok(())
}

/// Calculate number of rows based on data size string (e.g., "100MB", "1GB")
fn calculate_rows(data_size: &str, cols: i32, workers: i32) -> CylonResult<i64> {
    let size_str = data_size.to_uppercase();
    let (num_str, unit) = if size_str.ends_with("GB") {
        (&size_str[..size_str.len() - 2], 1_000_000_000i64)
    } else if size_str.ends_with("MB") {
        (&size_str[..size_str.len() - 2], 1_000_000i64)
    } else {
        return Err(cylon::error::CylonError::new(
            cylon::error::Code::Invalid,
            "Data size must end with MB or GB",
        ));
    };

    let size_num: i64 = num_str.parse().map_err(|_| {
        cylon::error::CylonError::new(cylon::error::Code::Invalid, "Invalid size number")
    })?;

    // Each value is int64 (8 bytes)
    Ok((size_num * unit) / (cols as i64 * workers as i64 * 8))
}
