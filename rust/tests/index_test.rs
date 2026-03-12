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

//! Tests for Table index operations
//!
//! Corresponds to C++ Table::SetArrowIndex, GetArrowIndex, ResetArrowIndex

use cylon::ctx::CylonContext;
use cylon::table::Table;
use cylon::indexing::{BaseArrowIndex, ArrowRangeIndex, ArrowLinearIndex, IndexingType};
use std::sync::Arc;
use arrow::array::{Array, Int32Array, Int64Array, StringArray};
use arrow::datatypes::{Schema, Field, DataType};
use arrow::record_batch::RecordBatch;

fn create_test_table(ctx: Arc<CylonContext>) -> Table {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("value", DataType::Int32, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])),
            Arc::new(StringArray::from(vec!["alice", "bob", "charlie", "david", "eve"])),
            Arc::new(Int32Array::from(vec![100, 200, 300, 400, 500])),
        ],
    ).unwrap();

    Table::from_record_batch(ctx, batch).unwrap()
}

#[test]
fn test_get_arrow_index_initially_none() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table(ctx.clone());

    assert!(table.get_arrow_index().is_none());

    println!("Get arrow index initially none test passed!");
}

#[test]
fn test_set_and_get_range_index() {
    let ctx = Arc::new(CylonContext::new(false));
    let mut table = create_test_table(ctx.clone());

    // Create and set a range index
    let range_index: Arc<dyn BaseArrowIndex> = Arc::new(ArrowRangeIndex::new(0, 5, 1));
    table.set_arrow_index(range_index.clone(), false).unwrap();

    // Get the index back
    let retrieved_index = table.get_arrow_index();
    assert!(retrieved_index.is_some());

    let index = retrieved_index.unwrap();
    assert_eq!(index.get_indexing_type(), IndexingType::Range);
    assert_eq!(index.get_size(), 5);

    println!("Set and get range index test passed!");
}

#[test]
fn test_set_index_with_drop() {
    let ctx = Arc::new(CylonContext::new(false));
    let mut table = create_test_table(ctx.clone());

    assert_eq!(table.columns(), 3);

    // Set index using first column (id) and drop it
    let index_arr = table.batch(0).unwrap().column(0).clone();
    let linear_index: Arc<dyn BaseArrowIndex> = Arc::new(ArrowLinearIndex::new(0, index_arr));

    table.set_arrow_index(linear_index, true).unwrap();

    // Table should now have 2 columns (id column was dropped)
    assert_eq!(table.columns(), 2);

    // Verify the index is set
    assert!(table.get_arrow_index().is_some());

    println!("Set index with drop test passed!");
}

#[test]
fn test_set_index_without_drop() {
    let ctx = Arc::new(CylonContext::new(false));
    let mut table = create_test_table(ctx.clone());

    assert_eq!(table.columns(), 3);

    // Set index using first column (id) without dropping
    let index_arr = table.batch(0).unwrap().column(0).clone();
    let linear_index: Arc<dyn BaseArrowIndex> = Arc::new(ArrowLinearIndex::new(0, index_arr));

    table.set_arrow_index(linear_index, false).unwrap();

    // Table should still have 3 columns
    assert_eq!(table.columns(), 3);

    // Verify the index is set
    assert!(table.get_arrow_index().is_some());

    println!("Set index without drop test passed!");
}

#[test]
fn test_reset_arrow_index_with_drop() {
    let ctx = Arc::new(CylonContext::new(false));
    let mut table = create_test_table(ctx.clone());

    // Set a linear index
    let index_arr = Arc::new(Int64Array::from(vec![10, 20, 30, 40, 50]));
    let linear_index: Arc<dyn BaseArrowIndex> = Arc::new(ArrowLinearIndex::new(0, index_arr));
    table.set_arrow_index(linear_index, false).unwrap();

    let original_cols = table.columns();

    // Reset index with drop=true
    table.reset_arrow_index(true).unwrap();

    // Should be converted to a range index
    let index = table.get_arrow_index().unwrap();
    assert_eq!(index.get_indexing_type(), IndexingType::Range);

    // Column count should remain the same (old index was dropped)
    assert_eq!(table.columns(), original_cols);

    println!("Reset arrow index with drop test passed!");
}

#[test]
fn test_reset_arrow_index_without_drop() {
    let ctx = Arc::new(CylonContext::new(false));
    let mut table = create_test_table(ctx.clone());

    // Set a linear index
    let index_arr = Arc::new(Int64Array::from(vec![10, 20, 30, 40, 50]));
    let linear_index: Arc<dyn BaseArrowIndex> = Arc::new(ArrowLinearIndex::new(0, index_arr));
    table.set_arrow_index(linear_index, false).unwrap();

    let original_cols = table.columns();

    // Reset index with drop=false
    table.reset_arrow_index(false).unwrap();

    // Should be converted to a range index
    let index = table.get_arrow_index().unwrap();
    assert_eq!(index.get_indexing_type(), IndexingType::Range);

    // Column count should increase by 1 (old index added as column)
    assert_eq!(table.columns(), original_cols + 1);

    // First column should be the old index
    let batch = table.batch(0).unwrap();
    assert_eq!(batch.schema().field(0).name(), "index");

    let index_col = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(index_col.value(0), 10);
    assert_eq!(index_col.value(4), 50);

    println!("Reset arrow index without drop test passed!");
}

#[test]
fn test_reset_range_index_no_op() {
    let ctx = Arc::new(CylonContext::new(false));
    let mut table = create_test_table(ctx.clone());

    // Set a range index
    let range_index: Arc<dyn BaseArrowIndex> = Arc::new(ArrowRangeIndex::new(0, 5, 1));
    table.set_arrow_index(range_index, false).unwrap();

    let original_cols = table.columns();

    // Reset should be a no-op for range index
    table.reset_arrow_index(false).unwrap();

    // Should still be a range index
    let index = table.get_arrow_index().unwrap();
    assert_eq!(index.get_indexing_type(), IndexingType::Range);

    // Column count should not change
    assert_eq!(table.columns(), original_cols);

    println!("Reset range index no-op test passed!");
}

#[test]
fn test_range_index_properties() {
    let range_index = ArrowRangeIndex::new(10, 5, 2);

    assert_eq!(range_index.get_size(), 5);
    assert_eq!(range_index.start(), 10);
    assert_eq!(range_index.step(), 2);
    assert_eq!(range_index.get_indexing_type(), IndexingType::Range);
    assert!(range_index.is_unique()); // step != 0

    // Get index as array
    let index_arr = range_index.get_index_array().unwrap();
    let int_arr = index_arr.as_any().downcast_ref::<Int64Array>().unwrap();

    assert_eq!(int_arr.len(), 5);
    assert_eq!(int_arr.value(0), 10);
    assert_eq!(int_arr.value(1), 12);
    assert_eq!(int_arr.value(2), 14);
    assert_eq!(int_arr.value(3), 16);
    assert_eq!(int_arr.value(4), 18);

    println!("Range index properties test passed!");
}

#[test]
fn test_linear_index_properties() {
    let index_arr = Arc::new(Int64Array::from(vec![5, 10, 15, 20]));
    let linear_index = ArrowLinearIndex::new(2, index_arr.clone());

    assert_eq!(linear_index.get_col_id(), 2);
    assert_eq!(linear_index.get_size(), 4);
    assert_eq!(linear_index.get_indexing_type(), IndexingType::Linear);

    // Get index array
    let retrieved_arr = linear_index.get_index_array().unwrap();
    let int_arr = retrieved_arr.as_any().downcast_ref::<Int64Array>().unwrap();

    assert_eq!(int_arr.len(), 4);
    assert_eq!(int_arr.value(0), 5);
    assert_eq!(int_arr.value(3), 20);

    println!("Linear index properties test passed!");
}

#[test]
fn test_set_index_invalid_column() {
    let ctx = Arc::new(CylonContext::new(false));
    let mut table = create_test_table(ctx.clone());

    // Try to set index with invalid column ID
    let index_arr = Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5]));
    let linear_index: Arc<dyn BaseArrowIndex> = Arc::new(ArrowLinearIndex::new(10, index_arr));

    let result = table.set_arrow_index(linear_index, true);

    assert!(result.is_err());
    if let Err(e) = result {
        let err_msg = e.to_string();
        assert!(err_msg.contains("out of bounds"));
    }

    println!("Set index invalid column test passed!");
}

#[test]
fn test_index_with_multi_batch_table() {
    let ctx = Arc::new(CylonContext::new(false));

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Int32, false),
    ]));

    let batch1 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(Int32Array::from(vec![10, 20, 30])),
        ],
    ).unwrap();

    let batch2 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![4, 5])),
            Arc::new(Int32Array::from(vec![40, 50])),
        ],
    ).unwrap();

    let mut table = Table::from_record_batches(ctx.clone(), vec![batch1, batch2]).unwrap();

    assert_eq!(table.num_batches(), 2);

    // Set index - should combine batches
    let range_index: Arc<dyn BaseArrowIndex> = Arc::new(ArrowRangeIndex::new(0, 5, 1));
    table.set_arrow_index(range_index, false).unwrap();

    // Should be combined into one batch
    assert_eq!(table.num_batches(), 1);
    assert_eq!(table.rows(), 5);

    println!("Index with multi-batch table test passed!");
}
