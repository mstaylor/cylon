// Tests for filter operations

use cylon_wasm::table::{TableData, ColumnData, Table};
use cylon_wasm::filter::{filter, filter_single, FilterConfig, Predicate, FilterValue, CompareOp, LogicOp};

fn create_test_table() -> Table {
    let mut data = TableData::new();
    data.add_column("id", ColumnData::Int32(vec![Some(1), Some(2), Some(3), Some(4), Some(5)])).unwrap();
    data.add_column("value", ColumnData::Float64(vec![
        Some(10.0), Some(25.0), Some(30.0), Some(45.0), Some(50.0)
    ])).unwrap();
    data.add_column("name", ColumnData::String(vec![
        Some("alice".to_string()),
        Some("bob".to_string()),
        Some("carol".to_string()),
        Some("dave".to_string()),
        Some("eve".to_string()),
    ])).unwrap();
    Table::from_table_data(&data).unwrap()
}

#[test]
fn test_filter_gt() {
    let table = create_test_table();
    let result = filter_single(&table, 1, CompareOp::Gt, FilterValue::Float(30.0)).unwrap();
    // value > 30: 45, 50 = 2 rows
    assert_eq!(result.num_rows(), 2);
}

#[test]
fn test_filter_ge() {
    let table = create_test_table();
    let result = filter_single(&table, 1, CompareOp::Ge, FilterValue::Float(30.0)).unwrap();
    // value >= 30: 30, 45, 50 = 3 rows
    assert_eq!(result.num_rows(), 3);
}

#[test]
fn test_filter_lt() {
    let table = create_test_table();
    let result = filter_single(&table, 1, CompareOp::Lt, FilterValue::Float(30.0)).unwrap();
    // value < 30: 10, 25 = 2 rows
    assert_eq!(result.num_rows(), 2);
}

#[test]
fn test_filter_le() {
    let table = create_test_table();
    let result = filter_single(&table, 1, CompareOp::Le, FilterValue::Float(30.0)).unwrap();
    // value <= 30: 10, 25, 30 = 3 rows
    assert_eq!(result.num_rows(), 3);
}

#[test]
fn test_filter_eq_int() {
    let table = create_test_table();
    let result = filter_single(&table, 0, CompareOp::Eq, FilterValue::Int(3)).unwrap();
    // id == 3: 1 row
    assert_eq!(result.num_rows(), 1);
}

#[test]
fn test_filter_ne_int() {
    let table = create_test_table();
    let result = filter_single(&table, 0, CompareOp::Ne, FilterValue::Int(3)).unwrap();
    // id != 3: 4 rows
    assert_eq!(result.num_rows(), 4);
}

#[test]
fn test_filter_eq_string() {
    let table = create_test_table();
    let result = filter_single(&table, 2, CompareOp::Eq, FilterValue::String("bob".to_string())).unwrap();
    assert_eq!(result.num_rows(), 1);
}

#[test]
fn test_filter_and_logic() {
    let table = create_test_table();

    let config = FilterConfig {
        predicates: vec![
            Predicate { column: 0, op: CompareOp::Gt, value: FilterValue::Int(1) },
            Predicate { column: 1, op: CompareOp::Lt, value: FilterValue::Float(50.0) },
        ],
        logic: LogicOp::And,
    };
    let result = filter(&table, &config).unwrap();

    // id > 1 AND value < 50: rows 2,3,4 (ids 2,3,4 with values 25,30,45)
    assert_eq!(result.num_rows(), 3);
}

#[test]
fn test_filter_or_logic() {
    let table = create_test_table();

    let config = FilterConfig {
        predicates: vec![
            Predicate { column: 0, op: CompareOp::Eq, value: FilterValue::Int(1) },
            Predicate { column: 0, op: CompareOp::Eq, value: FilterValue::Int(5) },
        ],
        logic: LogicOp::Or,
    };
    let result = filter(&table, &config).unwrap();

    // id == 1 OR id == 5: 2 rows
    assert_eq!(result.num_rows(), 2);
}

#[test]
fn test_filter_no_matches() {
    let table = create_test_table();
    let result = filter_single(&table, 0, CompareOp::Gt, FilterValue::Int(100)).unwrap();
    assert_eq!(result.num_rows(), 0);
}

#[test]
fn test_filter_all_match() {
    let table = create_test_table();
    let result = filter_single(&table, 0, CompareOp::Gt, FilterValue::Int(0)).unwrap();
    assert_eq!(result.num_rows(), 5);
}

#[test]
fn test_filter_empty_predicates() {
    let table = create_test_table();
    let config = FilterConfig::new(vec![]);
    let result = filter(&table, &config).unwrap();
    // No predicates = return all rows
    assert_eq!(result.num_rows(), 5);
}

#[test]
fn test_filter_column_out_of_bounds() {
    let table = create_test_table();
    let result = filter_single(&table, 10, CompareOp::Eq, FilterValue::Int(1));
    assert!(result.is_err());
}

#[test]
fn test_filter_with_nulls() {
    let mut data = TableData::new();
    data.add_column("value", ColumnData::Int32(vec![Some(1), None, Some(3), None, Some(5)])).unwrap();
    let table = Table::from_table_data(&data).unwrap();

    let result = filter_single(&table, 0, CompareOp::Gt, FilterValue::Int(0)).unwrap();
    // NULL values don't match predicates, so only 1, 3, 5 = 3 rows
    assert_eq!(result.num_rows(), 3);
}

#[test]
fn test_filter_preserves_columns() {
    let table = create_test_table();
    let result = filter_single(&table, 0, CompareOp::Eq, FilterValue::Int(1)).unwrap();

    // Should preserve all columns
    assert_eq!(result.num_columns(), 3);
}
