// Tests for groupby operations

use cylon_wasm::table::{TableData, ColumnData, Table};
use cylon_wasm::groupby::{hash_groupby, aggregate_column, GroupByConfig, Aggregation, AggregationOp};

fn create_test_table() -> Table {
    let mut data = TableData::new();
    data.add_column("category", ColumnData::String(vec![
        Some("A".to_string()),
        Some("B".to_string()),
        Some("A".to_string()),
        Some("B".to_string()),
        Some("A".to_string()),
    ])).unwrap();
    data.add_column("value", ColumnData::Float64(vec![
        Some(10.0),
        Some(20.0),
        Some(30.0),
        Some(40.0),
        Some(50.0),
    ])).unwrap();
    Table::from_table_data(&data).unwrap()
}

#[test]
fn test_groupby_sum() {
    let table = create_test_table();

    let config = GroupByConfig::new(
        vec![0],
        vec![Aggregation { column: 1, op: AggregationOp::Sum, alias: None }],
    );
    let result = hash_groupby(&table, &config).unwrap();

    // 2 groups: A and B
    assert_eq!(result.num_rows(), 2);
    // key column + aggregation column
    assert_eq!(result.num_columns(), 2);
}

#[test]
fn test_groupby_mean() {
    let table = create_test_table();

    let config = GroupByConfig::new(
        vec![0],
        vec![Aggregation { column: 1, op: AggregationOp::Mean, alias: None }],
    );
    let result = hash_groupby(&table, &config).unwrap();

    assert_eq!(result.num_rows(), 2);
}

#[test]
fn test_groupby_count() {
    let table = create_test_table();

    let config = GroupByConfig::new(
        vec![0],
        vec![Aggregation { column: 1, op: AggregationOp::Count, alias: None }],
    );
    let result = hash_groupby(&table, &config).unwrap();

    assert_eq!(result.num_rows(), 2);
}

#[test]
fn test_groupby_min_max() {
    let table = create_test_table();

    let config = GroupByConfig::new(
        vec![0],
        vec![
            Aggregation { column: 1, op: AggregationOp::Min, alias: None },
            Aggregation { column: 1, op: AggregationOp::Max, alias: None },
        ],
    );
    let result = hash_groupby(&table, &config).unwrap();

    assert_eq!(result.num_rows(), 2);
    assert_eq!(result.num_columns(), 3); // key + min + max
}

#[test]
fn test_groupby_multiple_aggregations() {
    let table = create_test_table();

    let config = GroupByConfig::new(
        vec![0],
        vec![
            Aggregation { column: 1, op: AggregationOp::Sum, alias: Some("total".to_string()) },
            Aggregation { column: 1, op: AggregationOp::Mean, alias: Some("average".to_string()) },
            Aggregation { column: 1, op: AggregationOp::Count, alias: Some("cnt".to_string()) },
        ],
    );
    let result = hash_groupby(&table, &config).unwrap();

    assert_eq!(result.num_columns(), 4); // key + 3 aggregations
}

#[test]
fn test_groupby_no_keys_error() {
    let table = create_test_table();

    let config = GroupByConfig::new(
        vec![],
        vec![Aggregation { column: 1, op: AggregationOp::Sum, alias: None }],
    );
    assert!(hash_groupby(&table, &config).is_err());
}

#[test]
fn test_aggregate_column_sum() {
    let table = create_test_table();
    let result = aggregate_column(&table, 1, AggregationOp::Sum).unwrap();
    // 10 + 20 + 30 + 40 + 50 = 150
    assert!((result - 150.0).abs() < 1e-6);
}

#[test]
fn test_aggregate_column_mean() {
    let table = create_test_table();
    let result = aggregate_column(&table, 1, AggregationOp::Mean).unwrap();
    // 150 / 5 = 30
    assert!((result - 30.0).abs() < 1e-6);
}

#[test]
fn test_aggregate_column_min() {
    let table = create_test_table();
    let result = aggregate_column(&table, 1, AggregationOp::Min).unwrap();
    assert!((result - 10.0).abs() < 1e-6);
}

#[test]
fn test_aggregate_column_max() {
    let table = create_test_table();
    let result = aggregate_column(&table, 1, AggregationOp::Max).unwrap();
    assert!((result - 50.0).abs() < 1e-6);
}

#[test]
fn test_aggregate_column_count() {
    let table = create_test_table();
    let result = aggregate_column(&table, 1, AggregationOp::Count).unwrap();
    assert!((result - 5.0).abs() < 1e-6);
}

#[test]
fn test_aggregate_column_out_of_bounds() {
    let table = create_test_table();
    assert!(aggregate_column(&table, 10, AggregationOp::Sum).is_err());
}

#[test]
fn test_groupby_with_int_keys() {
    let mut data = TableData::new();
    data.add_column("id", ColumnData::Int32(vec![Some(1), Some(2), Some(1), Some(2)])).unwrap();
    data.add_column("value", ColumnData::Float64(vec![Some(10.0), Some(20.0), Some(30.0), Some(40.0)])).unwrap();
    let table = Table::from_table_data(&data).unwrap();

    let config = GroupByConfig::new(
        vec![0],
        vec![Aggregation { column: 1, op: AggregationOp::Sum, alias: None }],
    );
    let result = hash_groupby(&table, &config).unwrap();

    assert_eq!(result.num_rows(), 2);
}
