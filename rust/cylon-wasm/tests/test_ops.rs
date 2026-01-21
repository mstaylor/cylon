// Tests for table operations (ops.rs)

use cylon_wasm::table::{TableData, ColumnData, Table};
use cylon_wasm::ops::{
    project, project_by_names,
    slice, head, tail,
    sort, merge,
    union, subtract, intersect, unique,
    hash_partition,
};

fn create_sample_table() -> Table {
    let mut data = TableData::new();
    data.add_column("id", ColumnData::Int32(vec![Some(1), Some(2), Some(3), Some(4), Some(5)])).unwrap();
    data.add_column("name", ColumnData::String(vec![
        Some("alice".to_string()),
        Some("bob".to_string()),
        Some("carol".to_string()),
        Some("dave".to_string()),
        Some("eve".to_string()),
    ])).unwrap();
    data.add_column("value", ColumnData::Float64(vec![
        Some(10.0), Some(20.0), Some(30.0), Some(40.0), Some(50.0)
    ])).unwrap();
    Table::from_table_data(&data).unwrap()
}

// =============================================================================
// Column Selection Tests
// =============================================================================

#[test]
fn test_project_single_column() {
    let table = create_sample_table();
    let result = project(&table, &[0]).unwrap();

    assert_eq!(result.num_columns(), 1);
    assert_eq!(result.num_rows(), 5);
}

#[test]
fn test_project_multiple_columns() {
    let table = create_sample_table();
    let result = project(&table, &[0, 2]).unwrap();

    assert_eq!(result.num_columns(), 2);
    assert_eq!(result.num_rows(), 5);
}

#[test]
fn test_project_reorder_columns() {
    let table = create_sample_table();
    let result = project(&table, &[2, 0, 1]).unwrap();

    assert_eq!(result.num_columns(), 3);
    let schema = result.schema();
    assert_eq!(schema.field(0).name(), "value");
    assert_eq!(schema.field(1).name(), "id");
    assert_eq!(schema.field(2).name(), "name");
}

#[test]
fn test_project_by_names() {
    let table = create_sample_table();
    let result = project_by_names(&table, &["name", "id"]).unwrap();

    assert_eq!(result.num_columns(), 2);
    let schema = result.schema();
    assert_eq!(schema.field(0).name(), "name");
    assert_eq!(schema.field(1).name(), "id");
}

// =============================================================================
// Row Selection Tests
// =============================================================================

#[test]
fn test_slice_middle() {
    let table = create_sample_table();
    let result = slice(&table, 1, 3).unwrap();

    assert_eq!(result.num_rows(), 3);
}

#[test]
fn test_slice_from_start() {
    let table = create_sample_table();
    let result = slice(&table, 0, 2).unwrap();

    assert_eq!(result.num_rows(), 2);
}

#[test]
fn test_head() {
    let table = create_sample_table();
    let result = head(&table, 3).unwrap();

    assert_eq!(result.num_rows(), 3);
}

#[test]
fn test_head_more_than_rows() {
    let table = create_sample_table();
    let result = head(&table, 10).unwrap();

    assert_eq!(result.num_rows(), 5); // Only 5 rows exist
}

#[test]
fn test_tail() {
    let table = create_sample_table();
    let result = tail(&table, 2).unwrap();

    assert_eq!(result.num_rows(), 2);
}

// =============================================================================
// Sort Tests
// =============================================================================

#[test]
fn test_sort_ascending() {
    let mut data = TableData::new();
    data.add_column("value", ColumnData::Int32(vec![Some(3), Some(1), Some(2)])).unwrap();
    let table = Table::from_table_data(&data).unwrap();

    let result = sort(&table, 0, true).unwrap();
    let result_data = result.to_table_data().unwrap();

    if let ColumnData::Int32(values) = &result_data.data[0] {
        assert_eq!(values, &vec![Some(1), Some(2), Some(3)]);
    } else {
        panic!("Wrong column type");
    }
}

#[test]
fn test_sort_descending() {
    let mut data = TableData::new();
    data.add_column("value", ColumnData::Int32(vec![Some(1), Some(3), Some(2)])).unwrap();
    let table = Table::from_table_data(&data).unwrap();

    let result = sort(&table, 0, false).unwrap();
    let result_data = result.to_table_data().unwrap();

    if let ColumnData::Int32(values) = &result_data.data[0] {
        assert_eq!(values, &vec![Some(3), Some(2), Some(1)]);
    } else {
        panic!("Wrong column type");
    }
}

// =============================================================================
// Merge Tests
// =============================================================================

#[test]
fn test_merge_two_tables() {
    let mut data1 = TableData::new();
    data1.add_column("id", ColumnData::Int32(vec![Some(1), Some(2)])).unwrap();
    let table1 = Table::from_table_data(&data1).unwrap();

    let mut data2 = TableData::new();
    data2.add_column("id", ColumnData::Int32(vec![Some(3), Some(4)])).unwrap();
    let table2 = Table::from_table_data(&data2).unwrap();

    let result = merge(&[&table1, &table2]).unwrap();

    assert_eq!(result.num_rows(), 4);
    assert_eq!(result.num_columns(), 1);
}

#[test]
fn test_merge_three_tables() {
    let mut data = TableData::new();
    data.add_column("x", ColumnData::Int32(vec![Some(1)])).unwrap();
    let t1 = Table::from_table_data(&data).unwrap();
    let t2 = Table::from_table_data(&data).unwrap();
    let t3 = Table::from_table_data(&data).unwrap();

    let result = merge(&[&t1, &t2, &t3]).unwrap();

    assert_eq!(result.num_rows(), 3);
}

// =============================================================================
// Set Operations Tests
// =============================================================================

#[test]
fn test_union() {
    let mut data1 = TableData::new();
    data1.add_column("id", ColumnData::Int32(vec![Some(1), Some(2)])).unwrap();
    let table1 = Table::from_table_data(&data1).unwrap();

    let mut data2 = TableData::new();
    data2.add_column("id", ColumnData::Int32(vec![Some(2), Some(3)])).unwrap();
    let table2 = Table::from_table_data(&data2).unwrap();

    let result = union(&table1, &table2).unwrap();

    // Union removes duplicates: {1, 2, 3}
    assert_eq!(result.num_rows(), 3);
}

#[test]
fn test_subtract() {
    let mut data1 = TableData::new();
    data1.add_column("id", ColumnData::Int32(vec![Some(1), Some(2), Some(3)])).unwrap();
    let table1 = Table::from_table_data(&data1).unwrap();

    let mut data2 = TableData::new();
    data2.add_column("id", ColumnData::Int32(vec![Some(2)])).unwrap();
    let table2 = Table::from_table_data(&data2).unwrap();

    let result = subtract(&table1, &table2).unwrap();

    // {1, 2, 3} - {2} = {1, 3}
    assert_eq!(result.num_rows(), 2);
}

#[test]
fn test_intersect() {
    let mut data1 = TableData::new();
    data1.add_column("id", ColumnData::Int32(vec![Some(1), Some(2), Some(3)])).unwrap();
    let table1 = Table::from_table_data(&data1).unwrap();

    let mut data2 = TableData::new();
    data2.add_column("id", ColumnData::Int32(vec![Some(2), Some(3), Some(4)])).unwrap();
    let table2 = Table::from_table_data(&data2).unwrap();

    let result = intersect(&table1, &table2).unwrap();

    // {1, 2, 3} ∩ {2, 3, 4} = {2, 3}
    assert_eq!(result.num_rows(), 2);
}

#[test]
fn test_unique() {
    let mut data = TableData::new();
    data.add_column("id", ColumnData::Int32(vec![Some(1), Some(1), Some(2), Some(2), Some(3)])).unwrap();
    data.add_column("value", ColumnData::Int32(vec![Some(10), Some(20), Some(30), Some(40), Some(50)])).unwrap();
    let table = Table::from_table_data(&data).unwrap();

    let result = unique(&table, &[0], true).unwrap();

    // Unique on column 0: {1, 2, 3}
    assert_eq!(result.num_rows(), 3);
}

// =============================================================================
// Hash Partition Tests
// =============================================================================

#[test]
fn test_hash_partition_creates_correct_number() {
    let table = create_sample_table();
    let partitions = hash_partition(&table, &[0], 4).unwrap();

    assert_eq!(partitions.len(), 4);
}

#[test]
fn test_hash_partition_preserves_total_rows() {
    let table = create_sample_table();
    let partitions = hash_partition(&table, &[0], 3).unwrap();

    let total_rows: usize = partitions.iter().map(|p| p.num_rows()).sum();
    assert_eq!(total_rows, 5);
}

#[test]
fn test_hash_partition_deterministic() {
    let table = create_sample_table();

    let partitions1 = hash_partition(&table, &[0], 4).unwrap();
    let partitions2 = hash_partition(&table, &[0], 4).unwrap();

    for (p1, p2) in partitions1.iter().zip(partitions2.iter()) {
        assert_eq!(p1.num_rows(), p2.num_rows());
    }
}

#[test]
fn test_hash_partition_single_partition() {
    let table = create_sample_table();
    let partitions = hash_partition(&table, &[0], 1).unwrap();

    assert_eq!(partitions.len(), 1);
    assert_eq!(partitions[0].num_rows(), 5);
}