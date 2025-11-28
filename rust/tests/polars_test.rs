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

//! Tests for Polars integration
//!
//! Tests zero-copy conversion between Cylon Table and Polars DataFrame
//! via the Arrow C Data Interface.

#[cfg(feature = "polars")]
mod polars_tests {
    use std::sync::Arc;
    use arrow::array::{Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use cylon::ctx::CylonContext;
    use cylon::Table;

    fn create_test_table() -> (Arc<CylonContext>, Table) {
        let ctx = Arc::new(CylonContext::new(false)); // false = local (non-distributed)

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("value", DataType::Int64, false),
        ]));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5])),
                Arc::new(StringArray::from(vec!["Alice", "Bob", "Charlie", "David", "Eve"])),
                Arc::new(Int64Array::from(vec![100, 200, 150, 300, 250])),
            ],
        ).unwrap();

        let table = Table::from_record_batch(ctx.clone(), batch).unwrap();
        (ctx, table)
    }

    #[test]
    fn test_to_polars() {
        let (_ctx, table) = create_test_table();

        // Convert to Polars DataFrame
        let df = table.to_polars().unwrap();

        // Verify the DataFrame has correct structure
        assert_eq!(df.width(), 3);
        assert_eq!(df.height(), 5);

        let columns: Vec<String> = df.get_column_names()
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(columns, vec!["id", "name", "value"]);
    }

    #[test]
    fn test_from_polars() {
        use polars::prelude::*;

        let (cylon_ctx, _) = create_test_table();

        // Create a Polars DataFrame directly
        let s1 = Column::new("x".into(), &[10i64, 20, 30]);
        let s2 = Column::new("y".into(), &[1i64, 2, 3]);

        let df = DataFrame::new(vec![s1, s2]).unwrap();

        // Convert to Cylon Table
        let table = Table::from_polars(cylon_ctx, &df).unwrap();

        // Verify the Table has correct structure
        assert_eq!(table.columns(), 2);
        assert_eq!(table.rows(), 3);
        assert_eq!(table.column_names(), vec!["x", "y"]);
    }

    #[test]
    fn test_round_trip() {
        let (ctx, original_table) = create_test_table();

        let original_rows = original_table.rows();
        let original_cols = original_table.columns();
        let original_names = original_table.column_names();

        // Convert to Polars DataFrame
        let df = original_table.to_polars().unwrap();

        // Convert back to Cylon Table
        let round_tripped = Table::from_polars(ctx.clone(), &df).unwrap();

        // Verify the data is preserved
        assert_eq!(round_tripped.rows(), original_rows);
        assert_eq!(round_tripped.columns(), original_cols);
        assert_eq!(round_tripped.column_names(), original_names);
    }

    #[test]
    fn test_polars_filter() {
        use polars::prelude::*;

        let (ctx, table) = create_test_table();

        // Convert to Polars DataFrame
        let df = table.to_polars().unwrap();

        // Apply a filter using Polars - compare column to a scalar
        let value_col = df.column("value").unwrap();
        let mask = value_col.as_materialized_series().gt(150).unwrap();
        let filtered = df.filter(&mask).unwrap();

        // Convert back to Cylon Table
        let result = Table::from_polars(ctx, &filtered).unwrap();

        // Should have filtered out rows where value <= 150
        // Original: [100, 200, 150, 300, 250] -> Filtered: [200, 300, 250]
        assert_eq!(result.rows(), 3);
    }

    #[test]
    fn test_polars_select() {
        let (ctx, table) = create_test_table();

        // Convert to Polars DataFrame
        let df = table.to_polars().unwrap();

        // Select only specific columns
        let selected = df.select(["name", "value"]).unwrap();

        // Convert back to Cylon Table
        let result = Table::from_polars(ctx, &selected).unwrap();

        // Should have only 2 columns now
        assert_eq!(result.columns(), 2);
        assert_eq!(result.rows(), 5);
        assert_eq!(result.column_names(), vec!["name", "value"]);
    }

    #[test]
    fn test_empty_table_conversion() {
        let ctx = Arc::new(CylonContext::new(false));

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
        ]));

        // Create empty batch
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from(Vec::<i64>::new()))],
        ).unwrap();

        let table = Table::from_record_batch(ctx.clone(), batch).unwrap();
        assert_eq!(table.rows(), 0);

        // Convert to Polars and back
        let df = table.to_polars().unwrap();
        let result = Table::from_polars(ctx, &df).unwrap();

        assert_eq!(result.rows(), 0);
        assert_eq!(result.columns(), 1);
    }
}

#[cfg(not(feature = "polars"))]
mod polars_disabled {
    #[test]
    fn polars_feature_not_enabled() {
        println!("Polars tests skipped - feature not enabled");
    }
}
