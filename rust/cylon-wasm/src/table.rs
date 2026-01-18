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

//! Table representation for WASM
//!
//! Two-layer design bridging Arrow computation with JSON interop.
//! See docs/TABLE_ARCHITECTURE.md for detailed explanation.

use std::sync::Arc;
use arrow::array::{
    ArrayRef, Int32Array, Int64Array, Float32Array, Float64Array,
    StringArray, BooleanArray, Array,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use serde::{Deserialize, Serialize};

use crate::error::{WasmError, WasmResult};

/// Column data in a serializable format for JSON interchange
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum ColumnData {
    Int32(Vec<Option<i32>>),
    Int64(Vec<Option<i64>>),
    Float32(Vec<Option<f32>>),
    Float64(Vec<Option<f64>>),
    String(Vec<Option<String>>),
    Boolean(Vec<Option<bool>>),
}

impl ColumnData {
    pub fn len(&self) -> usize {
        match self {
            ColumnData::Int32(v) => v.len(),
            ColumnData::Int64(v) => v.len(),
            ColumnData::Float32(v) => v.len(),
            ColumnData::Float64(v) => v.len(),
            ColumnData::String(v) => v.len(),
            ColumnData::Boolean(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Convert to Arrow array
    pub fn to_arrow_array(&self) -> ArrayRef {
        match self {
            ColumnData::Int32(v) => Arc::new(Int32Array::from(v.clone())) as ArrayRef,
            ColumnData::Int64(v) => Arc::new(Int64Array::from(v.clone())) as ArrayRef,
            ColumnData::Float32(v) => Arc::new(Float32Array::from(v.clone())) as ArrayRef,
            ColumnData::Float64(v) => Arc::new(Float64Array::from(v.clone())) as ArrayRef,
            ColumnData::String(v) => Arc::new(StringArray::from(v.clone())) as ArrayRef,
            ColumnData::Boolean(v) => Arc::new(BooleanArray::from(v.clone())) as ArrayRef,
        }
    }

    /// Create from Arrow array
    pub fn from_arrow_array(array: &ArrayRef) -> WasmResult<Self> {
        match array.data_type() {
            DataType::Int32 => {
                let arr = array.as_any().downcast_ref::<Int32Array>()
                    .ok_or_else(|| WasmError::type_error("Failed to downcast to Int32Array"))?;
                let values: Vec<Option<i32>> = (0..arr.len())
                    .map(|i| if arr.is_null(i) { None } else { Some(arr.value(i)) })
                    .collect();
                Ok(ColumnData::Int32(values))
            }
            DataType::Int64 => {
                let arr = array.as_any().downcast_ref::<Int64Array>()
                    .ok_or_else(|| WasmError::type_error("Failed to downcast to Int64Array"))?;
                let values: Vec<Option<i64>> = (0..arr.len())
                    .map(|i| if arr.is_null(i) { None } else { Some(arr.value(i)) })
                    .collect();
                Ok(ColumnData::Int64(values))
            }
            DataType::Float32 => {
                let arr = array.as_any().downcast_ref::<Float32Array>()
                    .ok_or_else(|| WasmError::type_error("Failed to downcast to Float32Array"))?;
                let values: Vec<Option<f32>> = (0..arr.len())
                    .map(|i| if arr.is_null(i) { None } else { Some(arr.value(i)) })
                    .collect();
                Ok(ColumnData::Float32(values))
            }
            DataType::Float64 => {
                let arr = array.as_any().downcast_ref::<Float64Array>()
                    .ok_or_else(|| WasmError::type_error("Failed to downcast to Float64Array"))?;
                let values: Vec<Option<f64>> = (0..arr.len())
                    .map(|i| if arr.is_null(i) { None } else { Some(arr.value(i)) })
                    .collect();
                Ok(ColumnData::Float64(values))
            }
            DataType::Utf8 => {
                let arr = array.as_any().downcast_ref::<StringArray>()
                    .ok_or_else(|| WasmError::type_error("Failed to downcast to StringArray"))?;
                let values: Vec<Option<String>> = (0..arr.len())
                    .map(|i| if arr.is_null(i) { None } else { Some(arr.value(i).to_string()) })
                    .collect();
                Ok(ColumnData::String(values))
            }
            DataType::Boolean => {
                let arr = array.as_any().downcast_ref::<BooleanArray>()
                    .ok_or_else(|| WasmError::type_error("Failed to downcast to BooleanArray"))?;
                let values: Vec<Option<bool>> = (0..arr.len())
                    .map(|i| if arr.is_null(i) { None } else { Some(arr.value(i)) })
                    .collect();
                Ok(ColumnData::Boolean(values))
            }
            dt => Err(WasmError::unsupported(format!("Unsupported data type: {:?}", dt))),
        }
    }

    pub fn data_type(&self) -> DataType {
        match self {
            ColumnData::Int32(_) => DataType::Int32,
            ColumnData::Int64(_) => DataType::Int64,
            ColumnData::Float32(_) => DataType::Float32,
            ColumnData::Float64(_) => DataType::Float64,
            ColumnData::String(_) => DataType::Utf8,
            ColumnData::Boolean(_) => DataType::Boolean,
        }
    }
}

/// Serializable table format for JSON interchange with JS/Python
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableData {
    pub columns: Vec<String>,
    pub data: Vec<ColumnData>,
}

impl TableData {
    pub fn new() -> Self {
        Self {
            columns: Vec::new(),
            data: Vec::new(),
        }
    }

    pub fn num_rows(&self) -> usize {
        self.data.first().map(|c| c.len()).unwrap_or(0)
    }

    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    pub fn add_column(&mut self, name: impl Into<String>, data: ColumnData) -> WasmResult<()> {
        if !self.data.is_empty() && data.len() != self.num_rows() {
            return Err(WasmError::invalid(
                format!("Column length {} doesn't match table rows {}", data.len(), self.num_rows())
            ));
        }
        self.columns.push(name.into());
        self.data.push(data);
        Ok(())
    }

    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|c| c == name)
    }

    pub fn column(&self, index: usize) -> Option<&ColumnData> {
        self.data.get(index)
    }

    pub fn column_by_name(&self, name: &str) -> Option<&ColumnData> {
        self.column_index(name).and_then(|i| self.column(i))
    }

    /// Convert to Arrow RecordBatch
    pub fn to_record_batch(&self) -> WasmResult<RecordBatch> {
        if self.columns.is_empty() {
            return Err(WasmError::invalid("Cannot create RecordBatch from empty table"));
        }

        let fields: Vec<Field> = self.columns.iter()
            .zip(self.data.iter())
            .map(|(name, col)| Field::new(name, col.data_type(), true))
            .collect();

        let schema = Arc::new(Schema::new(fields));
        let arrays: Vec<ArrayRef> = self.data.iter()
            .map(|col| col.to_arrow_array())
            .collect();

        RecordBatch::try_new(schema, arrays)
            .map_err(|e| WasmError::arrow_error(e.to_string()))
    }

    /// Create from Arrow RecordBatch
    pub fn from_record_batch(batch: &RecordBatch) -> WasmResult<Self> {
        let columns: Vec<String> = batch.schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();

        let data: WasmResult<Vec<ColumnData>> = batch.columns()
            .iter()
            .map(|col| ColumnData::from_arrow_array(col))
            .collect();

        Ok(Self { columns, data: data? })
    }

    pub fn from_json(json: &str) -> WasmResult<Self> {
        serde_json::from_str(json).map_err(WasmError::from)
    }

    pub fn to_json(&self) -> WasmResult<String> {
        serde_json::to_string(self).map_err(WasmError::from)
    }

    pub fn to_json_pretty(&self) -> WasmResult<String> {
        serde_json::to_string_pretty(self).map_err(WasmError::from)
    }
}

impl Default for TableData {
    fn default() -> Self {
        Self::new()
    }
}

/// Internal table wrapper holding Arrow RecordBatch
#[derive(Debug, Clone)]
pub struct Table {
    batch: RecordBatch,
}

impl Table {
    pub fn new(batch: RecordBatch) -> Self {
        Self { batch }
    }

    pub fn from_table_data(data: &TableData) -> WasmResult<Self> {
        let batch = data.to_record_batch()?;
        Ok(Self { batch })
    }

    pub fn to_table_data(&self) -> WasmResult<TableData> {
        TableData::from_record_batch(&self.batch)
    }

    pub fn batch(&self) -> &RecordBatch {
        &self.batch
    }

    pub fn num_rows(&self) -> usize {
        self.batch.num_rows()
    }

    pub fn num_columns(&self) -> usize {
        self.batch.num_columns()
    }

    pub fn schema(&self) -> Arc<Schema> {
        self.batch.schema()
    }

    pub fn column(&self, index: usize) -> Option<&ArrayRef> {
        if index < self.batch.num_columns() {
            Some(self.batch.column(index))
        } else {
            None
        }
    }

    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.batch.schema().index_of(name).ok()
    }

    pub fn column_by_name(&self, name: &str) -> Option<&ArrayRef> {
        self.column_index(name).and_then(|i| self.column(i))
    }

    /// Serialize table to Arrow IPC format (binary)
    pub fn to_arrow_ipc(&self) -> WasmResult<Vec<u8>> {
        cylon::net::serialize::serialize_record_batch(&self.batch)
            .map_err(|e| WasmError::execution_error(e.to_string()))
    }

    /// Deserialize table from Arrow IPC format (binary)
    pub fn from_arrow_ipc(data: &[u8]) -> WasmResult<Self> {
        let batch = cylon::net::serialize::deserialize_record_batch(data)
            .map_err(|e| WasmError::execution_error(e.to_string()))?;
        Ok(Self { batch })
    }
}
