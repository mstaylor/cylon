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

//! Tests for table and column serialization
//!
//! Tests for cylon::net::serialize module

use std::sync::Arc;
use arrow::array::{Array, Int32Array, Int64Array, Float64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use cylon::ctx::CylonContext;
use cylon::table::Table;
use cylon::net::serialize::{
    serialize_record_batch, deserialize_record_batch,
    serialize_table, deserialize_table,
    CylonColumnSerializer, ColumnSerializer,
};

fn create_test_batch() -> RecordBatch {
    let schema = Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Utf8, false),
    ]);

    let a = Int32Array::from(vec![1, 2, 3, 4, 5]);
    let b = StringArray::from(vec!["a", "b", "c", "d", "e"]);

    RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(a), Arc::new(b)],
    ).unwrap()
}

#[test]
fn test_serialize_deserialize_record_batch() {
    let batch = create_test_batch();

    // Serialize
    let bytes = serialize_record_batch(&batch).unwrap();
    assert!(!bytes.is_empty());

    // Deserialize
    let result = deserialize_record_batch(&bytes).unwrap();

    // Verify schema
    assert_eq!(result.schema(), batch.schema());
    assert_eq!(result.num_rows(), batch.num_rows());
    assert_eq!(result.num_columns(), batch.num_columns());

    // Verify data
    let a_orig = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    let a_result = result.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    assert_eq!(a_orig.len(), a_result.len());
    for i in 0..a_orig.len() {
        assert_eq!(a_orig.value(i), a_result.value(i));
    }

    let b_orig = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();
    let b_result = result.column(1).as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(b_orig.len(), b_result.len());
    for i in 0..b_orig.len() {
        assert_eq!(b_orig.value(i), b_result.value(i));
    }
}

#[test]
fn test_serialize_deserialize_table() {
    let ctx = Arc::new(CylonContext::new(false));

    // Create a table with multiple batches
    let batch1 = create_test_batch();
    let batch2 = create_test_batch();

    let table = Table::from_record_batches(
        ctx.clone(),
        vec![batch1, batch2],
    ).unwrap();

    // Serialize
    let bytes = serialize_table(&table).unwrap();
    assert!(!bytes.is_empty());

    // Deserialize
    let result = deserialize_table(ctx, &bytes).unwrap();

    // Verify
    assert_eq!(result.num_batches(), table.num_batches());
    assert_eq!(result.rows(), table.rows());
    assert_eq!(result.columns(), table.columns());
    assert_eq!(result.schema(), table.schema());
}

#[test]
fn test_empty_table() {
    let ctx = Arc::new(CylonContext::new(false));

    // Create an empty batch with schema
    let schema = Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Utf8, false),
    ]);

    let a = Int32Array::from(vec![] as Vec<i32>);
    let b = StringArray::from(vec![] as Vec<&str>);

    let empty_batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(a), Arc::new(b)],
    ).unwrap();

    let table = Table::from_record_batches(
        ctx.clone(),
        vec![empty_batch],
    ).unwrap();

    // Serialize
    let bytes = serialize_table(&table).unwrap();

    // Deserialize
    let result = deserialize_table(ctx, &bytes).unwrap();

    // Verify
    assert_eq!(result.num_batches(), 1);
    assert_eq!(result.rows(), 0); // No rows but has schema
}

// =============================================================================
// ColumnSerializer Tests
// =============================================================================

#[test]
fn test_column_serializer_int64() {
    let array: Arc<dyn Array> = Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5]));
    let serializer = CylonColumnSerializer::make(&array).unwrap();

    let sizes = serializer.buffer_sizes();
    assert_eq!(sizes[0], 0, "No null buffer for non-nullable array");
    assert_eq!(sizes[1], 0, "No offset buffer for fixed-width type");
    assert_eq!(sizes[2], 40, "5 * 8 bytes = 40 bytes for int64");
}

#[test]
fn test_column_serializer_float64() {
    let array: Arc<dyn Array> = Arc::new(Float64Array::from(vec![1.1, 2.2, 3.3]));
    let serializer = CylonColumnSerializer::make(&array).unwrap();

    let sizes = serializer.buffer_sizes();
    assert_eq!(sizes[0], 0, "No null buffer for non-nullable array");
    assert_eq!(sizes[1], 0, "No offset buffer for fixed-width type");
    assert_eq!(sizes[2], 24, "3 * 8 bytes = 24 bytes for float64");
}

#[test]
fn test_column_serializer_string() {
    let array: Arc<dyn Array> = Arc::new(StringArray::from(vec!["hello", "world"]));
    let serializer = CylonColumnSerializer::make(&array).unwrap();

    let sizes = serializer.buffer_sizes();
    assert_eq!(sizes[0], 0, "No null buffer for non-nullable array");
    assert_eq!(sizes[1], 12, "3 offsets * 4 bytes = 12 bytes for string offsets");
    assert_eq!(sizes[2], 10, "5 + 5 = 10 bytes for string data");
}

#[test]
fn test_column_serializer_empty() {
    let array: Arc<dyn Array> = Arc::new(Int64Array::from(vec![] as Vec<i64>));
    let serializer = CylonColumnSerializer::make(&array).unwrap();

    let sizes = serializer.buffer_sizes();
    assert_eq!(sizes[0], 0);
    assert_eq!(sizes[1], 0);
    assert_eq!(sizes[2], 0);
}
