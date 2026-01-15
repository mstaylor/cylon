// Tests for join operations

use cylon_wasm::table::{TableData, ColumnData, Table};
use cylon_wasm::join::{hash_join, JoinConfig, JoinType};

fn create_left_table() -> Table {
    let mut data = TableData::new();
    data.add_column("id", ColumnData::Int32(vec![Some(1), Some(2), Some(3)])).unwrap();
    data.add_column("name", ColumnData::String(vec![
        Some("alice".to_string()),
        Some("bob".to_string()),
        Some("carol".to_string()),
    ])).unwrap();
    Table::from_table_data(&data).unwrap()
}

fn create_right_table() -> Table {
    let mut data = TableData::new();
    data.add_column("id", ColumnData::Int32(vec![Some(1), Some(2), Some(4)])).unwrap();
    data.add_column("dept", ColumnData::String(vec![
        Some("eng".to_string()),
        Some("sales".to_string()),
        Some("marketing".to_string()),
    ])).unwrap();
    Table::from_table_data(&data).unwrap()
}

#[test]
fn test_inner_join() {
    let left = create_left_table();
    let right = create_right_table();

    let config = JoinConfig::new(JoinType::Inner, vec![0], vec![0]);
    let result = hash_join(&left, &right, &config).unwrap();

    // Inner: only matching rows (id=1, id=2)
    assert_eq!(result.num_rows(), 2);
    assert_eq!(result.num_columns(), 4);
}

#[test]
fn test_left_join() {
    let left = create_left_table();
    let right = create_right_table();

    let config = JoinConfig::new(JoinType::Left, vec![0], vec![0]);
    let result = hash_join(&left, &right, &config).unwrap();

    // Left: all left rows (3)
    assert_eq!(result.num_rows(), 3);
}

#[test]
fn test_right_join() {
    let left = create_left_table();
    let right = create_right_table();

    let config = JoinConfig::new(JoinType::Right, vec![0], vec![0]);
    let result = hash_join(&left, &right, &config).unwrap();

    // Right: all right rows (3)
    assert_eq!(result.num_rows(), 3);
}

#[test]
fn test_full_outer_join() {
    let left = create_left_table();
    let right = create_right_table();

    let config = JoinConfig::new(JoinType::FullOuter, vec![0], vec![0]);
    let result = hash_join(&left, &right, &config).unwrap();

    // Full outer: 2 matched + 1 left-only + 1 right-only = 4
    assert_eq!(result.num_rows(), 4);
}

#[test]
fn test_join_column_suffixes() {
    let left = create_left_table();
    let right = create_right_table();

    let mut config = JoinConfig::new(JoinType::Inner, vec![0], vec![0]);
    config.left_suffix = "_left".to_string();
    config.right_suffix = "_right".to_string();

    let result = hash_join(&left, &right, &config).unwrap();
    let schema = result.schema();
    let names: Vec<String> = schema.fields().iter()
        .map(|f| f.name().clone()).collect();

    assert!(names.iter().any(|n| n == "id_left"));
    assert!(names.iter().any(|n| n == "id_right"));
}

#[test]
fn test_join_mismatched_column_count() {
    let left = create_left_table();
    let right = create_right_table();

    let config = JoinConfig::new(JoinType::Inner, vec![0, 1], vec![0]);
    assert!(hash_join(&left, &right, &config).is_err());
}

#[test]
fn test_join_empty_columns() {
    let left = create_left_table();
    let right = create_right_table();

    let config = JoinConfig::new(JoinType::Inner, vec![], vec![]);
    assert!(hash_join(&left, &right, &config).is_err());
}

#[test]
fn test_join_with_duplicates() {
    let mut left_data = TableData::new();
    left_data.add_column("id", ColumnData::Int32(vec![Some(1), Some(1), Some(2)])).unwrap();
    left_data.add_column("value", ColumnData::Int32(vec![Some(10), Some(20), Some(30)])).unwrap();
    let left = Table::from_table_data(&left_data).unwrap();

    let mut right_data = TableData::new();
    right_data.add_column("id", ColumnData::Int32(vec![Some(1), Some(2)])).unwrap();
    right_data.add_column("label", ColumnData::String(vec![
        Some("a".to_string()),
        Some("b".to_string()),
    ])).unwrap();
    let right = Table::from_table_data(&right_data).unwrap();

    let config = JoinConfig::new(JoinType::Inner, vec![0], vec![0]);
    let result = hash_join(&left, &right, &config).unwrap();

    // id=1 matches twice, id=2 matches once = 3 rows
    assert_eq!(result.num_rows(), 3);
}

#[test]
fn test_join_no_matches() {
    let mut left_data = TableData::new();
    left_data.add_column("id", ColumnData::Int32(vec![Some(1), Some(2)])).unwrap();
    let left = Table::from_table_data(&left_data).unwrap();

    let mut right_data = TableData::new();
    right_data.add_column("id", ColumnData::Int32(vec![Some(3), Some(4)])).unwrap();
    let right = Table::from_table_data(&right_data).unwrap();

    let config = JoinConfig::new(JoinType::Inner, vec![0], vec![0]);
    let result = hash_join(&left, &right, &config).unwrap();

    assert_eq!(result.num_rows(), 0);
}
