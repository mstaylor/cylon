//! Incremental checkpoint support.
//!
//! This module provides change tracking and delta-based checkpointing to reduce
//! checkpoint size and time by only writing modified data.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::RwLock;

/// Tracks changes since the last checkpoint for incremental support.
///
/// The `ChangeTracker` monitors which tables have been modified and optionally
/// tracks row-level changes for fine-grained incremental checkpoints.
pub struct ChangeTracker {
    /// Tables that have been modified since last checkpoint
    modified_tables: RwLock<HashSet<String>>,

    /// For row-level tracking: modified row ranges per table
    modified_ranges: RwLock<HashMap<String, Vec<RowRange>>>,

    /// Parent checkpoint this is based on
    parent_checkpoint_id: RwLock<Option<u64>>,

    /// Whether to track row-level changes (more overhead but smaller deltas)
    track_rows: bool,
}

impl ChangeTracker {
    /// Create a new change tracker
    pub fn new() -> Self {
        Self {
            modified_tables: RwLock::new(HashSet::new()),
            modified_ranges: RwLock::new(HashMap::new()),
            parent_checkpoint_id: RwLock::new(None),
            track_rows: false,
        }
    }

    /// Create a change tracker with row-level tracking enabled
    pub fn with_row_tracking() -> Self {
        Self {
            modified_tables: RwLock::new(HashSet::new()),
            modified_ranges: RwLock::new(HashMap::new()),
            parent_checkpoint_id: RwLock::new(None),
            track_rows: true,
        }
    }

    /// Create a change tracker based on a parent checkpoint
    pub fn from_checkpoint(parent_checkpoint_id: u64) -> Self {
        Self {
            modified_tables: RwLock::new(HashSet::new()),
            modified_ranges: RwLock::new(HashMap::new()),
            parent_checkpoint_id: RwLock::new(Some(parent_checkpoint_id)),
            track_rows: false,
        }
    }

    /// Get the parent checkpoint ID
    pub fn parent_checkpoint_id(&self) -> Option<u64> {
        *self.parent_checkpoint_id.read().unwrap()
    }

    /// Set the parent checkpoint ID (called after a checkpoint is created)
    pub fn set_parent_checkpoint(&self, checkpoint_id: u64) {
        let mut parent = self.parent_checkpoint_id.write().unwrap();
        *parent = Some(checkpoint_id);
    }

    /// Mark a table as modified (full table change)
    pub fn mark_table_modified(&self, table_name: &str) {
        let mut tables = self.modified_tables.write().unwrap();
        tables.insert(table_name.to_string());
    }

    /// Mark specific rows as modified (for fine-grained incremental)
    pub fn mark_rows_modified(&self, table_name: &str, range: RowRange) {
        // Always mark the table as modified
        {
            let mut tables = self.modified_tables.write().unwrap();
            tables.insert(table_name.to_string());
        }

        // If row tracking is enabled, record the specific range
        if self.track_rows {
            let mut ranges = self.modified_ranges.write().unwrap();
            ranges
                .entry(table_name.to_string())
                .or_default()
                .push(range);
        }
    }

    /// Mark rows as appended to a table
    pub fn mark_rows_appended(&self, table_name: &str, start_row: u64, count: u64) {
        self.mark_rows_modified(
            table_name,
            RowRange {
                start: start_row,
                end: start_row + count,
                change_type: RowChangeType::Append,
            },
        );
    }

    /// Check if a table has been modified since the last checkpoint
    pub fn is_table_modified(&self, table_name: &str) -> bool {
        let tables = self.modified_tables.read().unwrap();
        tables.contains(table_name)
    }

    /// Check if a table needs to be checkpointed
    ///
    /// Returns true if:
    /// - There's no parent checkpoint (first checkpoint, must write all)
    /// - The table has been modified since the last checkpoint
    pub fn needs_checkpoint(&self, table_name: &str) -> bool {
        let parent = self.parent_checkpoint_id.read().unwrap();
        if parent.is_none() {
            // No parent checkpoint - must write everything
            return true;
        }
        self.is_table_modified(table_name)
    }

    /// Get the modified row ranges for a table
    pub fn get_modified_ranges(&self, table_name: &str) -> Vec<RowRange> {
        let ranges = self.modified_ranges.read().unwrap();
        ranges.get(table_name).cloned().unwrap_or_default()
    }

    /// Get all modified table names
    pub fn get_modified_tables(&self) -> Vec<String> {
        let tables = self.modified_tables.read().unwrap();
        tables.iter().cloned().collect()
    }

    /// Get all unmodified table names (given a list of all tables)
    pub fn get_unchanged_tables(&self, all_tables: &[String]) -> Vec<String> {
        let tables = self.modified_tables.read().unwrap();
        all_tables
            .iter()
            .filter(|t| !tables.contains(*t))
            .cloned()
            .collect()
    }

    /// Reset the tracker after a checkpoint is created
    pub fn reset(&self) {
        let mut tables = self.modified_tables.write().unwrap();
        tables.clear();

        let mut ranges = self.modified_ranges.write().unwrap();
        ranges.clear();
    }

    /// Check if row tracking is enabled
    pub fn is_row_tracking_enabled(&self) -> bool {
        self.track_rows
    }

    /// Determine the delta type for a table based on tracked changes
    pub fn get_delta_type(&self, table_name: &str) -> DeltaType {
        if !self.track_rows {
            // Without row tracking, treat all changes as full rewrites
            return DeltaType::Full;
        }

        let ranges = self.modified_ranges.read().unwrap();
        let table_ranges = match ranges.get(table_name) {
            Some(r) => r,
            None => return DeltaType::Full, // No specific ranges tracked
        };

        if table_ranges.is_empty() {
            return DeltaType::Full;
        }

        // Check what types of changes occurred
        let has_append = table_ranges
            .iter()
            .any(|r| r.change_type == RowChangeType::Append);
        let has_update = table_ranges
            .iter()
            .any(|r| r.change_type == RowChangeType::Update);
        let has_delete = table_ranges
            .iter()
            .any(|r| r.change_type == RowChangeType::Delete);

        match (has_append, has_update, has_delete) {
            (true, false, false) => DeltaType::Append,
            (false, true, false) => DeltaType::Update,
            (false, false, true) => DeltaType::Delete,
            _ => DeltaType::Mixed,
        }
    }
}

impl Default for ChangeTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// A range of rows that have been modified.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RowRange {
    /// Start row index (inclusive)
    pub start: u64,
    /// End row index (exclusive)
    pub end: u64,
    /// Type of change
    pub change_type: RowChangeType,
}

impl RowRange {
    /// Create a new row range
    pub fn new(start: u64, end: u64, change_type: RowChangeType) -> Self {
        Self {
            start,
            end,
            change_type,
        }
    }

    /// Create an append range
    pub fn append(start: u64, end: u64) -> Self {
        Self::new(start, end, RowChangeType::Append)
    }

    /// Create an update range
    pub fn update(start: u64, end: u64) -> Self {
        Self::new(start, end, RowChangeType::Update)
    }

    /// Create a delete range
    pub fn delete(start: u64, end: u64) -> Self {
        Self::new(start, end, RowChangeType::Delete)
    }

    /// Get the number of rows in this range
    pub fn len(&self) -> u64 {
        self.end.saturating_sub(self.start)
    }

    /// Check if the range is empty
    pub fn is_empty(&self) -> bool {
        self.end <= self.start
    }
}

/// Type of row change.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RowChangeType {
    /// New rows appended
    Append,
    /// Existing rows updated
    Update,
    /// Rows deleted
    Delete,
}

/// Type of delta in an incremental checkpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeltaType {
    /// Append new rows to the table
    Append,
    /// Update existing rows (includes row indices)
    Update,
    /// Delete rows (includes row indices to remove)
    Delete,
    /// Mixed operations (append + update + delete)
    Mixed,
    /// Full table rewrite (no delta, complete replacement)
    Full,
}

/// Information about a table's delta in an incremental checkpoint.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeltaTableInfo {
    /// Table name
    pub name: String,
    /// Type of delta operation
    pub delta_type: DeltaType,
    /// Number of rows affected
    pub affected_rows: u64,
    /// For append: starting row index in the original table
    pub append_start_row: Option<u64>,
    /// For update/delete: affected row indices
    pub affected_indices: Option<Vec<u64>>,
}

impl DeltaTableInfo {
    /// Create info for an append delta
    pub fn append(name: impl Into<String>, rows: u64, start_row: u64) -> Self {
        Self {
            name: name.into(),
            delta_type: DeltaType::Append,
            affected_rows: rows,
            append_start_row: Some(start_row),
            affected_indices: None,
        }
    }

    /// Create info for a full table write
    pub fn full(name: impl Into<String>, rows: u64) -> Self {
        Self {
            name: name.into(),
            delta_type: DeltaType::Full,
            affected_rows: rows,
            append_start_row: None,
            affected_indices: None,
        }
    }

    /// Create info for an update delta
    pub fn update(name: impl Into<String>, indices: Vec<u64>) -> Self {
        let rows = indices.len() as u64;
        Self {
            name: name.into(),
            delta_type: DeltaType::Update,
            affected_rows: rows,
            append_start_row: None,
            affected_indices: Some(indices),
        }
    }

    /// Create info for a delete delta
    pub fn delete(name: impl Into<String>, indices: Vec<u64>) -> Self {
        let rows = indices.len() as u64;
        Self {
            name: name.into(),
            delta_type: DeltaType::Delete,
            affected_rows: rows,
            append_start_row: None,
            affected_indices: Some(indices),
        }
    }
}

/// Information about an incremental checkpoint.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IncrementalCheckpointInfo {
    /// The checkpoint ID this is based on
    pub parent_checkpoint_id: u64,

    /// Tables that are unchanged (reference parent)
    pub unchanged_tables: Vec<String>,

    /// Tables that have deltas
    pub delta_tables: Vec<DeltaTableInfo>,

    /// Tables that are fully rewritten (no delta possible)
    pub full_tables: Vec<String>,

    /// Depth of the checkpoint chain (1 = based on full checkpoint)
    pub chain_depth: u32,
}

impl IncrementalCheckpointInfo {
    /// Create new incremental checkpoint info
    pub fn new(parent_checkpoint_id: u64) -> Self {
        Self {
            parent_checkpoint_id,
            unchanged_tables: Vec::new(),
            delta_tables: Vec::new(),
            full_tables: Vec::new(),
            chain_depth: 1,
        }
    }

    /// Add an unchanged table
    pub fn add_unchanged(&mut self, table_name: impl Into<String>) {
        self.unchanged_tables.push(table_name.into());
    }

    /// Add a delta table
    pub fn add_delta(&mut self, info: DeltaTableInfo) {
        self.delta_tables.push(info);
    }

    /// Add a full table
    pub fn add_full(&mut self, table_name: impl Into<String>) {
        self.full_tables.push(table_name.into());
    }

    /// Get total number of tables in this checkpoint
    pub fn total_tables(&self) -> usize {
        self.unchanged_tables.len() + self.delta_tables.len() + self.full_tables.len()
    }

    /// Check if this is purely incremental (no full tables)
    pub fn is_pure_incremental(&self) -> bool {
        self.full_tables.is_empty()
    }

    /// Get the space savings ratio compared to a full checkpoint
    /// Returns a value between 0.0 (no savings) and 1.0 (maximum savings)
    pub fn savings_ratio(&self) -> f64 {
        let total = self.total_tables();
        if total == 0 {
            return 0.0;
        }
        self.unchanged_tables.len() as f64 / total as f64
    }
}

/// Result of restoring from an incremental checkpoint.
#[derive(Debug)]
pub struct IncrementalRestoreResult {
    /// The checkpoint ID that was restored
    pub checkpoint_id: u64,

    /// Chain of checkpoint IDs that were applied (oldest first)
    pub checkpoint_chain: Vec<u64>,

    /// Number of deltas applied
    pub deltas_applied: usize,

    /// Tables that were restored
    pub tables_restored: Vec<String>,
}

/// Configuration for incremental checkpoint behavior.
#[derive(Clone, Debug)]
pub struct IncrementalConfig {
    /// Enable incremental checkpoints
    pub enabled: bool,

    /// Enable row-level change tracking (more overhead, smaller deltas)
    pub track_rows: bool,

    /// Maximum chain depth before forcing a full checkpoint
    /// This prevents restore from becoming too slow
    pub max_chain_depth: u32,

    /// Force full checkpoint if savings ratio is below this threshold
    /// (i.e., if most tables are modified, just do a full checkpoint)
    pub min_savings_ratio: f64,
}

impl Default for IncrementalConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            track_rows: false,
            max_chain_depth: 10,
            min_savings_ratio: 0.2, // At least 20% unchanged tables to use incremental
        }
    }
}

impl IncrementalConfig {
    /// Create config with incremental enabled
    pub fn enabled() -> Self {
        Self {
            enabled: true,
            ..Default::default()
        }
    }

    /// Enable row-level tracking
    pub fn with_row_tracking(mut self) -> Self {
        self.track_rows = true;
        self
    }

    /// Set maximum chain depth
    pub fn with_max_chain_depth(mut self, depth: u32) -> Self {
        self.max_chain_depth = depth;
        self
    }

    /// Set minimum savings ratio
    pub fn with_min_savings_ratio(mut self, ratio: f64) -> Self {
        self.min_savings_ratio = ratio.clamp(0.0, 1.0);
        self
    }
}
