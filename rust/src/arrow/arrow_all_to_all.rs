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

//! Arrow table all-to-all communication with buffer-by-buffer transmission
//!
//! Ported from cpp/src/cylon/arrow/arrow_all_to_all.cpp
//!
//! This implementation sends Arrow table buffers individually with metadata,
//! allowing for memory-efficient streaming of large tables.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{Array, ArrayData, ArrayRef, make_array};
use arrow::buffer::Buffer;
use arrow::datatypes::{Schema, Field, DataType};
use arrow::record_batch::RecordBatch;
use arrow::compute::concat;

use crate::error::{CylonError, CylonResult, Code};
use crate::net::Buffer as NetBuffer;
use crate::net::ops::{AllToAll, ReceiveCallback};
use crate::table::Table;
use crate::ctx::CylonContext;

/// Callback for receiving Arrow tables
/// Corresponds to C++ ArrowCallback
pub type ArrowCallback = Box<dyn FnMut(i32, Table, i32) -> bool + Send + Sync>;

/// Arrow header constants
/// Corresponds to C++ ArrowHeader enum
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArrowHeader {
    Init = 0,
    ColumnContinue = 1,
    ColumnEnd = 2,
}

/// Pending table to send with buffer tracking
/// Corresponds to C++ PendingSendTable
struct PendingSendTable {
    target: i32,
    pending: Vec<(Table, i32)>, // (table, reference)
    current_table: Option<(Table, i32)>,
    status: ArrowHeader,
    column_index: usize,
    array_index: usize,
    buffer_index: usize,
}

impl PendingSendTable {
    fn new(target: i32) -> Self {
        Self {
            target,
            pending: Vec::new(),
            current_table: None,
            status: ArrowHeader::Init,
            column_index: 0,
            array_index: 0,
            buffer_index: 0,
        }
    }
}

/// Pending table being received with buffer reconstruction
/// Corresponds to C++ PendingReceiveTable
struct PendingReceiveTable {
    source: i32,
    column_index: usize,
    buffer_index: usize,
    no_buffers: usize,
    no_arrays: usize,
    length: usize,
    reference: i32,
    current_arrays: Vec<ArrayRef>,
    buffers: Vec<Option<Buffer>>,
    arrays: Vec<ArrayRef>,
}

impl PendingReceiveTable {
    fn new(source: i32) -> Self {
        Self {
            source,
            column_index: 0,
            buffer_index: 0,
            no_buffers: 0,
            no_arrays: 0,
            length: 0,
            reference: 0,
            current_arrays: Vec::new(),
            buffers: Vec::new(),
            arrays: Vec::new(),
        }
    }
}

// Wrapper type to implement ReceiveCallback using raw pointer to ArrowAllToAll
// This matches the C++ design where ArrowAllToAll passes `this` to AllToAll
struct ArrowAllToAllRecvWrapper(*mut ArrowAllToAll);
unsafe impl Send for ArrowAllToAllRecvWrapper {}
unsafe impl Sync for ArrowAllToAllRecvWrapper {}

impl ReceiveCallback for ArrowAllToAllRecvWrapper {
    fn on_receive(&mut self, source: i32, buffer: Box<dyn NetBuffer>, length: usize) -> bool {
        unsafe { (*self.0).on_receive(source, buffer, length) }
    }

    fn on_receive_header(&mut self, source: i32, finished: i32, header: Option<Vec<i32>>) -> bool {
        unsafe { (*self.0).on_receive_header(source, finished, header) }
    }

    fn on_send_complete(&mut self, target: i32, buffer: &[u8], length: usize) -> bool {
        unsafe { (*self.0).on_send_complete(target, buffer, length) }
    }
}

/// Arrow table all-to-all communication
///
/// Sends tables buffer-by-buffer for memory efficiency.
/// Corresponds to C++ ArrowAllToAll from cpp/src/cylon/arrow/arrow_all_to_all.hpp
pub struct ArrowAllToAll {
    targets: Vec<i32>,
    sources: Vec<i32>,
    all: Option<Box<AllToAll>>,  // Option because we need to create it after boxing self
    inputs: HashMap<i32, PendingSendTable>,
    receives: HashMap<i32, PendingReceiveTable>,
    recv_callback: Option<ArrowCallback>,
    schema: Arc<Schema>,
    ctx: Arc<CylonContext>,
    finished: bool,
    completed: bool,
    finish_called: bool,
    finished_sources: Vec<i32>,
    received_buffers: usize,
}

impl ArrowAllToAll {
    /// Create a new ArrowAllToAll operation
    ///
    /// Matches C++ ArrowAllToAll constructor (arrow_all_to_all.cpp:26-54)
    /// C++: all_ = std::make_shared<AllToAll>(ctx, source, targets, edgeId, this, allocator_);
    pub fn new(
        worker_id: i32,
        sources: Vec<i32>,
        targets: Vec<i32>,
        edge_id: i32,
        callback: ArrowCallback,
        schema: Arc<Schema>,
        ctx: Arc<CylonContext>,
        channel: Box<dyn crate::net::Channel>,
        allocator: Box<dyn crate::net::Allocator>,
    ) -> CylonResult<Box<Self>> {
        let mut inputs = HashMap::new();
        for &t in &targets {
            inputs.insert(t, PendingSendTable::new(t));
        }

        let mut receives = HashMap::new();
        for &s in &sources {
            receives.insert(s, PendingReceiveTable::new(s));
        }

        // Box first to get stable address (matching C++ heap allocation)
        let mut arrow_all_to_all = Box::new(Self {
            targets: targets.clone(),
            sources: sources.clone(),
            all: None,  // Will be set after creating AllToAll
            inputs,
            receives,
            recv_callback: Some(callback),
            schema,
            ctx,
            finished: false,
            completed: false,
            finish_called: false,
            finished_sources: Vec::new(),
            received_buffers: 0,
        });

        // Create AllToAll with self as callback (matches C++ pattern)
        // C++: all_ = std::make_shared<AllToAll>(ctx, source, targets, edgeId, this, allocator_);
        // Use raw pointer wrapper for callback
        unsafe {
            let self_ptr = &mut *arrow_all_to_all as *mut Self;
            let callback_wrapper: Box<dyn ReceiveCallback> = Box::new(ArrowAllToAllRecvWrapper(self_ptr));

            let all_to_all = AllToAll::new(
                worker_id,
                sources,
                targets,
                edge_id,
                channel,
                callback_wrapper,
                allocator,
            )?;

            arrow_all_to_all.all = Some(all_to_all);
        }

        Ok(arrow_all_to_all)
    }

    /// Insert a table to be sent to a target
    pub fn insert(&mut self, table: Table, target: i32) -> i32 {
        self.insert_with_reference(table, target, -1)
    }

    /// Insert a table with a reference value to be sent to a target
    pub fn insert_with_reference(&mut self, table: Table, target: i32, reference: i32) -> i32 {
        if let Some(pending) = self.inputs.get_mut(&target) {
            pending.pending.push((table, reference));
            1
        } else {
            -1
        }
    }

    // Helper to get AllToAll (panics if not initialized)
    fn all(&self) -> &AllToAll {
        self.all.as_ref().expect("AllToAll not initialized")
    }

    fn all_mut(&mut self) -> &mut AllToAll {
        self.all.as_mut().expect("AllToAll not initialized")
    }

    /// Check if the all-to-all operation is complete
    ///
    /// This sends buffers from pending tables and progresses the underlying AllToAll.
    /// Corresponds to C++ ArrowAllToAll::isComplete() (lines 70-152)
    pub fn is_complete(&mut self) -> CylonResult<bool> {
        if self.completed {
            return Ok(true);
        }

        let mut is_all_empty = true;

        // Get mutable reference to AllToAll before the loop to avoid borrow checker issues
        let all = self.all.as_mut().expect("AllToAll not initialized");

        // Send buffers for each target
        // This follows the C++ logic from lines 76-143
        for (&target, pending) in &mut self.inputs {
            // Check if we need to load a new table
            if pending.status == ArrowHeader::Init {
                if !pending.pending.is_empty() {
                    pending.current_table = Some(pending.pending.remove(0));
                    pending.status = ArrowHeader::ColumnContinue;
                }
            }

            // Send buffers from current table
            if pending.status == ArrowHeader::ColumnContinue {
                if let Some((ref table, reference)) = pending.current_table {
                    let batches = table.batches();
                    if batches.is_empty() {
                        // No data, move to next
                        pending.status = ArrowHeader::Init;
                        pending.column_index = 0;
                        continue;
                    }

                    let no_columns = batches[0].num_columns();
                    let mut can_continue = true;

                    while pending.column_index < no_columns && can_continue {
                        // MATERIAL DIFFERENCE: Data structure organization
                        // C++: Table has Columns (ChunkedArray), each ChunkedArray has chunks (Array)
                        //      Iteration: for each column { for each chunk { send buffers } }
                        // Rust: Table has Batches (RecordBatch), each RecordBatch has columns (Array)
                        //       Iteration: for each column { for each batch { send buffers } }
                        // Result is functionally equivalent - both send all array chunks for each column

                        // Get the column array from all batches
                        for (batch_idx, batch) in batches.iter().enumerate() {
                            if pending.array_index != batch_idx {
                                continue;
                            }

                            let column = batch.column(pending.column_index);
                            let array_data = column.to_data();
                            let buffers = array_data.buffers();

                            // Send each buffer in the array
                            while pending.buffer_index < buffers.len() {
                                let buf = &buffers[pending.buffer_index];

                                // Create header: [columnIndex, bufferIndex, noBuffers, noArrays, length, reference]
                                let header = [
                                    pending.column_index as i32,
                                    pending.buffer_index as i32,
                                    buffers.len() as i32,
                                    batches.len() as i32,
                                    array_data.len() as i32,
                                    reference,
                                ];

                                // Send the buffer
                                let result = if buf.is_empty() {
                                    // Send empty buffer
                                    all.insert_with_header(Vec::new(), target, &header)
                                } else {
                                    all.insert_with_header(buf.as_slice().to_vec(), target, &header)
                                };

                                if result == -1 {
                                    // Can't send right now
                                    can_continue = false;
                                    break;
                                }

                                pending.buffer_index += 1;
                            }

                            // If we sent all buffers for this array
                            if can_continue {
                                pending.buffer_index = 0;
                                pending.array_index += 1;
                            }

                            break; // Move to next batch
                        }

                        // If we sent all arrays for this column
                        if can_continue && pending.array_index >= batches.len() {
                            pending.array_index = 0;
                            pending.column_index += 1;
                        }
                    }

                    // If we sent everything for this table
                    if can_continue && pending.column_index >= no_columns {
                        pending.column_index = 0;
                        pending.array_index = 0;
                        pending.buffer_index = 0;
                        pending.status = ArrowHeader::Init;
                        pending.current_table = None;
                    }
                }
            }

            // Check if this target still has work
            if !pending.pending.is_empty() || pending.status == ArrowHeader::ColumnContinue {
                is_all_empty = false;
            }
        }

        // Check if we should call finish
        if is_all_empty && self.finished && !self.finish_called {
            all.finish();
            self.finish_called = true;
        }

        // Check completion
        let all_complete = all.is_complete();
        self.completed = is_all_empty
            && all_complete
            && self.finished_sources.len() == self.sources.len();

        Ok(self.completed)
    }

    /// Signal that no more tables will be inserted
    pub fn finish(&mut self) {
        self.finished = true;
    }

    /// Close the operation
    pub fn close(&mut self) {
        self.inputs.clear();
        self.all_mut().close();
    }
}

impl ReceiveCallback for ArrowAllToAll {
    /// Receive a buffer and reconstruct table
    /// Corresponds to C++ ArrowAllToAll::onReceive() (lines 174-213)
    fn on_receive(&mut self, source: i32, buffer: Box<dyn NetBuffer>, length: usize) -> bool {
        if let Some(pending) = self.receives.get_mut(&source) {
            self.received_buffers += 1;

            // Convert buffer to Arrow Buffer
            // MATERIAL DIFFERENCE: C++ checks if buffer[0] is empty AND HasValidityBitmap, then sets to nullptr
            // Rust Arrow's ArrayData builder handles null buffers differently
            // We store None for empty buffers, converted to empty Buffer later
            let data = buffer.get_byte_buffer()[..length].to_vec();
            let arrow_buffer = if data.is_empty() {
                None
            } else {
                Some(Buffer::from_vec(data))
            };

            pending.buffers.push(arrow_buffer);

            // Check if we have all buffers for this array
            if pending.no_buffers == pending.buffer_index + 1 {
                // Create ArrayData from collected buffers
                let field = &self.schema.fields()[pending.column_index];
                let data_type = field.data_type();

                // Handle null bitmap
                let mut buffers_vec = Vec::new();
                for buf in &pending.buffers {
                    if let Some(b) = buf {
                        buffers_vec.push(b.clone());
                    } else {
                        // Null buffer - create empty buffer
                        buffers_vec.push(Buffer::from(&[]));
                    }
                }

                // Create ArrayData
                let array_data = ArrayData::builder(data_type.clone())
                    .len(pending.length)
                    .buffers(buffers_vec)
                    .build()
                    .expect("Failed to build ArrayData");

                pending.buffers.clear();

                // Create Array
                let array = make_array(array_data);
                pending.arrays.push(array);

                // Check if we have all arrays for this column (chunks of ChunkedArray)
                if pending.arrays.len() == pending.no_arrays {
                    // MATERIAL DIFFERENCE: C++ creates ChunkedArray, Rust must concatenate
                    // C++ keeps multiple chunks as ChunkedArray, but Rust RecordBatch uses single arrays
                    // So we concatenate all array chunks into one array
                    let column_array = if pending.arrays.len() == 1 {
                        // Single array, no concatenation needed
                        pending.arrays.remove(0)
                    } else {
                        // Multiple arrays (chunks), must concatenate them
                        // This matches C++ ChunkedArray behavior but flattens to single array
                        let array_refs: Vec<&dyn Array> = pending.arrays.iter()
                            .map(|a| a.as_ref())
                            .collect();

                        match concat(&array_refs) {
                            Ok(concatenated) => concatenated,
                            Err(e) => {
                                eprintln!("Failed to concatenate array chunks: {}", e);
                                return false;
                            }
                        }
                    };

                    pending.current_arrays.push(column_array);
                    pending.arrays.clear();

                    // Check if we have all columns
                    if pending.current_arrays.len() == self.schema.fields().len() {
                        // Create RecordBatch and Table
                        match RecordBatch::try_new(
                            self.schema.clone(),
                            pending.current_arrays.clone(),
                        ) {
                            Ok(batch) => {
                                match Table::from_record_batches(self.ctx.clone(), vec![batch]) {
                                    Ok(table) => {
                                        pending.current_arrays.clear();

                                        // Call user callback
                                        if let Some(ref mut callback) = self.recv_callback {
                                            callback(source, table, pending.reference);
                                        }
                                    }
                                    Err(e) => {
                                        eprintln!("Failed to create Table: {}", e);
                                        return false;
                                    }
                                }
                            }
                            Err(e) => {
                                eprintln!("Failed to create RecordBatch: {}", e);
                                return false;
                            }
                        }
                    }
                }
            }
        }

        true
    }

    /// Receive header with buffer metadata
    /// Corresponds to C++ ArrowAllToAll::onReceiveHeader() (lines 215-233)
    fn on_receive_header(&mut self, source: i32, finished: i32, header: Option<Vec<i32>>) -> bool {
        if finished == 0 {
            // Not a finish message - extract buffer metadata
            if let Some(hdr) = header {
                if hdr.len() != 6 {
                    eprintln!("Incorrect header length, expected 6 got {}", hdr.len());
                    return false;
                }

                if let Some(pending) = self.receives.get_mut(&source) {
                    pending.column_index = hdr[0] as usize;
                    pending.buffer_index = hdr[1] as usize;
                    pending.no_buffers = hdr[2] as usize;
                    pending.no_arrays = hdr[3] as usize;
                    pending.length = hdr[4] as usize;
                    pending.reference = hdr[5];
                }
            }
        } else {
            // Finish message
            self.finished_sources.push(source);
        }

        true
    }

    fn on_send_complete(&mut self, _target: i32, _buffer: &[u8], _length: usize) -> bool {
        // Nothing to do on send complete
        true
    }
}
