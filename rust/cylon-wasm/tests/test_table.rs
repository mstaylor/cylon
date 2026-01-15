// Tests for table operations

use cylon_wasm::table::{TableData, ColumnData, Table};

#[test]
fn test_table_data_creation() {
    let mut table = TableData::new();
    table.add_column("id", ColumnData::Int32(vec![Some(1), Some(2), Some(3)])).unwrap();
    table.add_column("value", ColumnData::Float64(vec![Some(1.1), Some(2.2), Some(3.3)])).unwrap();

    assert_eq!(table.num_rows(), 3);
    assert_eq!(table.num_columns(), 2);
}

#[test]
fn test_table_data_with_nulls() {
    let mut table = TableData::new();
    table.add_column("id", ColumnData::Int32(vec![Some(1), None, Some(3)])).unwrap();
    assert_eq!(table.num_rows(), 3);
}

#[test]
fn test_table_data_column_length_mismatch() {
    let mut table = TableData::new();
    table.add_column("id", ColumnData::Int32(vec![Some(1), Some(2), Some(3)])).unwrap();
    let result = table.add_column("value", ColumnData::Float64(vec![Some(1.1), Some(2.2)]));
    assert!(result.is_err());
}

#[test]
fn test_table_data_json_round_trip() {
    let mut table = TableData::new();
    table.add_column("name", ColumnData::String(vec![
        Some("alice".to_string()),
        Some("bob".to_string())
    ])).unwrap();
    table.add_column("value", ColumnData::Int64(vec![Some(100), Some(200)])).unwrap();

    let json = table.to_json().unwrap();
    let parsed = TableData::from_json(&json).unwrap();

    assert_eq!(parsed.num_rows(), 2);
    assert_eq!(parsed.num_columns(), 2);
    assert_eq!(parsed.columns, table.columns);
}

#[test]
fn test_table_data_json_with_nulls() {
    let mut table = TableData::new();
    table.add_column("value", ColumnData::Int32(vec![Some(1), None, Some(3)])).unwrap();

    let json = table.to_json().unwrap();
    assert!(json.contains("null"));

    let parsed = TableData::from_json(&json).unwrap();
    if let ColumnData::Int32(values) = &parsed.data[0] {
        assert_eq!(values[0], Some(1));
        assert_eq!(values[1], None);
        assert_eq!(values[2], Some(3));
    } else {
        panic!("Wrong column type");
    }
}

#[test]
fn test_table_data_to_record_batch() {
    let mut table = TableData::new();
    table.add_column("id", ColumnData::Int32(vec![Some(1), Some(2)])).unwrap();
    table.add_column("flag", ColumnData::Boolean(vec![Some(true), Some(false)])).unwrap();

    let batch = table.to_record_batch().unwrap();
    assert_eq!(batch.num_rows(), 2);
    assert_eq!(batch.num_columns(), 2);
}

#[test]
fn test_table_wrapper() {
    let mut table_data = TableData::new();
    table_data.add_column("x", ColumnData::Float32(vec![Some(1.0), Some(2.0)])).unwrap();

    let table = Table::from_table_data(&table_data).unwrap();
    assert_eq!(table.num_rows(), 2);
    assert_eq!(table.num_columns(), 1);

    let back = table.to_table_data().unwrap();
    assert_eq!(back.num_rows(), 2);
}

#[test]
fn test_column_index_lookup() {
    let mut table = TableData::new();
    table.add_column("id", ColumnData::Int32(vec![Some(1)])).unwrap();
    table.add_column("name", ColumnData::String(vec![Some("test".to_string())])).unwrap();
    table.add_column("value", ColumnData::Float64(vec![Some(1.5)])).unwrap();

    assert_eq!(table.column_index("id"), Some(0));
    assert_eq!(table.column_index("name"), Some(1));
    assert_eq!(table.column_index("value"), Some(2));
    assert_eq!(table.column_index("nonexistent"), None);
}

#[test]
fn test_all_column_types() {
    let mut table = TableData::new();
    table.add_column("int32", ColumnData::Int32(vec![Some(1)])).unwrap();
    table.add_column("int64", ColumnData::Int64(vec![Some(2)])).unwrap();
    table.add_column("float32", ColumnData::Float32(vec![Some(3.0)])).unwrap();
    table.add_column("float64", ColumnData::Float64(vec![Some(4.0)])).unwrap();
    table.add_column("string", ColumnData::String(vec![Some("test".to_string())])).unwrap();
    table.add_column("bool", ColumnData::Boolean(vec![Some(true)])).unwrap();

    let batch = table.to_record_batch().unwrap();
    let back = TableData::from_record_batch(&batch).unwrap();
    assert_eq!(back.num_columns(), 6);
}
