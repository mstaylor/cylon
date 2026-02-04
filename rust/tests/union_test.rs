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

//! Tests for Union operation
//!
//! Corresponds to C++ Union function in table.cpp

use cylon::ctx::CylonContext;
use cylon::table::{Table, union};
use std::sync::Arc;
use arrow::array::{Array, Int32Array, StringArray};
use arrow::datatypes::{Schema, Field, DataType};
use arrow::record_batch::RecordBatch;
use std::collections::HashSet;

fn create_test_table(ctx: Arc<CylonContext>, ids: Vec<i32>, names: Vec<&str>) -> Table {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(StringArray::from(names)),
        ],
    ).unwrap();

    Table::from_record_batch(ctx, batch).unwrap()
}

#[test]
fn test_union_no_overlap() {
    let ctx = Arc::new(CylonContext::new(false));

    // Table 1: {(1, "a"), (2, "b")}
    let table1 = create_test_table(ctx.clone(), vec![1, 2], vec!["a", "b"]);

    // Table 2: {(3, "c"), (4, "d")}
    let table2 = create_test_table(ctx.clone(), vec![3, 4], vec!["c", "d"]);

    let result = union(&table1, &table2).unwrap();

    // Should have all 4 rows since there's no overlap
    assert_eq!(result.rows(), 4, "Union with no overlap should have 4 rows");
    assert_eq!(result.columns(), 2);

    // Verify all IDs are present
    let batch = result.batch(0).unwrap();
    let ids = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    let id_set: HashSet<i32> = (0..ids.len()).map(|i| ids.value(i)).collect();

    assert!(id_set.contains(&1));
    assert!(id_set.contains(&2));
    assert!(id_set.contains(&3));
    assert!(id_set.contains(&4));

    println!("Union no overlap test passed!");
}

#[test]
fn test_union_complete_overlap() {
    let ctx = Arc::new(CylonContext::new(false));

    // Both tables have the same data: {(1, "a"), (2, "b")}
    let table1 = create_test_table(ctx.clone(), vec![1, 2], vec!["a", "b"]);
    let table2 = create_test_table(ctx.clone(), vec![1, 2], vec!["a", "b"]);

    let result = union(&table1, &table2).unwrap();

    // Should have only 2 rows since all rows are duplicates
    assert_eq!(result.rows(), 2, "Union with complete overlap should have 2 rows");

    let batch = result.batch(0).unwrap();
    let ids = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    assert_eq!(ids.value(0), 1);
    assert_eq!(ids.value(1), 2);

    println!("Union complete overlap test passed!");
}

#[test]
fn test_union_partial_overlap() {
    let ctx = Arc::new(CylonContext::new(false));

    // Table 1: {(1, "a"), (2, "b"), (3, "c")}
    let table1 = create_test_table(ctx.clone(), vec![1, 2, 3], vec!["a", "b", "c"]);

    // Table 2: {(2, "b"), (3, "c"), (4, "d")}
    let table2 = create_test_table(ctx.clone(), vec![2, 3, 4], vec!["b", "c", "d"]);

    let result = union(&table1, &table2).unwrap();

    // Should have 4 unique rows: (1,a), (2,b), (3,c), (4,d)
    assert_eq!(result.rows(), 4, "Union with partial overlap should have 4 rows");

    let batch = result.batch(0).unwrap();
    let ids = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    let id_set: HashSet<i32> = (0..ids.len()).map(|i| ids.value(i)).collect();

    assert_eq!(id_set.len(), 4);
    assert!(id_set.contains(&1));
    assert!(id_set.contains(&2));
    assert!(id_set.contains(&3));
    assert!(id_set.contains(&4));

    println!("Union partial overlap test passed!");
}

#[test]
fn test_union_empty_first_table() {
    let ctx = Arc::new(CylonContext::new(false));

    let empty_table = create_test_table(ctx.clone(), vec![], vec![]);
    let table2 = create_test_table(ctx.clone(), vec![1, 2], vec!["a", "b"]);

    let result = union(&empty_table, &table2).unwrap();

    // Should have 2 rows from the non-empty table
    assert_eq!(result.rows(), 2);

    println!("Union empty first table test passed!");
}

#[test]
fn test_union_empty_second_table() {
    let ctx = Arc::new(CylonContext::new(false));

    let table1 = create_test_table(ctx.clone(), vec![1, 2], vec!["a", "b"]);
    let empty_table = create_test_table(ctx.clone(), vec![], vec![]);

    let result = union(&table1, &empty_table).unwrap();

    // Should have 2 rows from the non-empty table
    assert_eq!(result.rows(), 2);

    println!("Union empty second table test passed!");
}

#[test]
fn test_union_both_empty() {
    let ctx = Arc::new(CylonContext::new(false));

    let empty1 = create_test_table(ctx.clone(), vec![], vec![]);
    let empty2 = create_test_table(ctx.clone(), vec![], vec![]);

    let result = union(&empty1, &empty2).unwrap();

    assert_eq!(result.rows(), 0);
    assert_eq!(result.columns(), 2);

    println!("Union both empty test passed!");
}

#[test]
fn test_union_incompatible_schemas() {
    let ctx = Arc::new(CylonContext::new(false));

    // Table 1 with id and name
    let schema1 = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
    ]));

    let batch1 = RecordBatch::try_new(
        schema1,
        vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(StringArray::from(vec!["a", "b"])),
        ],
    ).unwrap();

    let table1 = Table::from_record_batch(ctx.clone(), batch1).unwrap();

    // Table 2 with different schema (different column names)
    let schema2 = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Utf8, false),  // Different column name
    ]));

    let batch2 = RecordBatch::try_new(
        schema2,
        vec![
            Arc::new(Int32Array::from(vec![3, 4])),
            Arc::new(StringArray::from(vec!["c", "d"])),
        ],
    ).unwrap();

    let table2 = Table::from_record_batch(ctx.clone(), batch2).unwrap();

    // Union should fail due to schema mismatch
    let result = union(&table1, &table2);
    assert!(result.is_err(), "Should fail with incompatible schemas");

    println!("Union incompatible schemas test passed!");
}

#[test]
fn test_union_duplicate_rows_in_same_table() {
    let ctx = Arc::new(CylonContext::new(false));

    // Table 1 has duplicates: {(1, "a"), (1, "a"), (2, "b")}
    let table1 = create_test_table(ctx.clone(), vec![1, 1, 2], vec!["a", "a", "b"]);

    // Table 2: {(3, "c")}
    let table2 = create_test_table(ctx.clone(), vec![3], vec!["c"]);

    let result = union(&table1, &table2).unwrap();

    // Union removes duplicates from first table too, so:
    // From table1: (1,"a") appears once, (2,"b")
    // From table2: (3,"c")
    // Total: 3 unique rows
    assert_eq!(result.rows(), 3, "Union should remove duplicates within same table");

    let batch = result.batch(0).unwrap();
    let ids = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    let id_set: HashSet<i32> = (0..ids.len()).map(|i| ids.value(i)).collect();

    assert_eq!(id_set.len(), 3);
    assert!(id_set.contains(&1));
    assert!(id_set.contains(&2));
    assert!(id_set.contains(&3));

    println!("Union duplicate rows in same table test passed!");
}

#[test]
fn test_union_preserves_schema() {
    let ctx = Arc::new(CylonContext::new(false));

    let table1 = create_test_table(ctx.clone(), vec![1, 2], vec!["a", "b"]);
    let table2 = create_test_table(ctx.clone(), vec![3, 4], vec!["c", "d"]);

    let result = union(&table1, &table2).unwrap();

    // Verify schema is preserved
    let orig_names = table1.column_names();
    let result_names = result.column_names();
    assert_eq!(orig_names, result_names);

    println!("Union preserves schema test passed!");
}

#[test]
fn test_union_large_tables() {
    let ctx = Arc::new(CylonContext::new(false));

    // Create larger tables with some overlap
    let ids1: Vec<i32> = (0..100).collect();
    let names1: Vec<String> = ids1.iter().map(|i| format!("name{}", i)).collect();
    let name_refs1: Vec<&str> = names1.iter().map(|s| s.as_str()).collect();

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
    ]));

    let batch1 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(ids1)),
            Arc::new(StringArray::from(name_refs1)),
        ],
    ).unwrap();

    // Second table overlaps: 50-149 (overlap on 50-99)
    let ids2: Vec<i32> = (50..150).collect();
    let names2: Vec<String> = ids2.iter().map(|i| format!("name{}", i)).collect();
    let name_refs2: Vec<&str> = names2.iter().map(|s| s.as_str()).collect();

    let batch2 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(ids2)),
            Arc::new(StringArray::from(name_refs2)),
        ],
    ).unwrap();

    let table1 = Table::from_record_batch(ctx.clone(), batch1).unwrap();
    let table2 = Table::from_record_batch(ctx.clone(), batch2).unwrap();

    let result = union(&table1, &table2).unwrap();

    // Should have 150 unique rows (0-149)
    assert_eq!(result.rows(), 150);

    println!("Union large tables test passed!");
}

#[test]
fn test_union_single_row_tables() {
    let ctx = Arc::new(CylonContext::new(false));

    let table1 = create_test_table(ctx.clone(), vec![1], vec!["a"]);
    let table2 = create_test_table(ctx.clone(), vec![2], vec!["b"]);

    let result = union(&table1, &table2).unwrap();

    assert_eq!(result.rows(), 2);

    let batch = result.batch(0).unwrap();
    let ids = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    let id_set: HashSet<i32> = (0..ids.len()).map(|i| ids.value(i)).collect();

    assert!(id_set.contains(&1));
    assert!(id_set.contains(&2));

    println!("Union single row tables test passed!");
}
