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

//! Tests for DataFusion integration
//!
//! Tests zero-copy conversion between Cylon Table and DataFusion DataFrame.

#[cfg(feature = "datafusion")]
mod datafusion_tests {
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

    #[tokio::test]
    async fn test_to_datafusion() {
        let (_ctx, table) = create_test_table();

        // Convert to DataFusion DataFrame
        let df = table.to_datafusion().await.unwrap();

        // Verify the DataFrame has correct structure
        let schema = df.schema();
        assert_eq!(schema.fields().len(), 3);
        assert_eq!(schema.field(0).name(), "id");
        assert_eq!(schema.field(1).name(), "name");
        assert_eq!(schema.field(2).name(), "value");
    }

    #[tokio::test]
    async fn test_from_datafusion() {
        use datafusion::prelude::*;

        let (cylon_ctx, _) = create_test_table();

        // Create a DataFusion DataFrame directly
        let session_ctx = SessionContext::new();

        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Int64, false),
            Field::new("y", DataType::Int64, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![10, 20, 30])),
                Arc::new(Int64Array::from(vec![1, 2, 3])),
            ],
        ).unwrap();

        let mem_table = datafusion::datasource::MemTable::try_new(
            schema, vec![vec![batch]]
        ).unwrap();

        session_ctx.register_table("test", Arc::new(mem_table)).unwrap();
        let df = session_ctx.table("test").await.unwrap();

        // Convert to Cylon Table
        let table = Table::from_datafusion(cylon_ctx, df).await.unwrap();

        // Verify the Table has correct structure
        assert_eq!(table.columns(), 2);
        assert_eq!(table.rows(), 3);
        assert_eq!(table.column_names(), vec!["x", "y"]);
    }

    #[tokio::test]
    async fn test_round_trip() {
        let (ctx, original_table) = create_test_table();

        let original_rows = original_table.rows();
        let original_cols = original_table.columns();
        let original_names = original_table.column_names();

        // Convert to DataFusion DataFrame
        let df = original_table.to_datafusion().await.unwrap();

        // Convert back to Cylon Table
        let round_tripped = Table::from_datafusion(ctx.clone(), df).await.unwrap();

        // Verify the data is preserved
        assert_eq!(round_tripped.rows(), original_rows);
        assert_eq!(round_tripped.columns(), original_cols);
        assert_eq!(round_tripped.column_names(), original_names);
    }

    #[tokio::test]
    async fn test_datafusion_filter() {
        use datafusion::prelude::*;

        let (ctx, table) = create_test_table();

        // Convert to DataFusion DataFrame
        let df = table.to_datafusion().await.unwrap();

        // Apply a filter using DataFusion
        let filtered = df.filter(col("value").gt(lit(150))).unwrap();

        // Convert back to Cylon Table
        let result = Table::from_datafusion(ctx, filtered).await.unwrap();

        // Should have filtered out rows where value <= 150
        // Original: [100, 200, 150, 300, 250] -> Filtered: [200, 300, 250]
        assert_eq!(result.rows(), 3);
    }

    #[tokio::test]
    async fn test_datafusion_select() {
        use datafusion::prelude::*;

        let (ctx, table) = create_test_table();

        // Convert to DataFusion DataFrame
        let df = table.to_datafusion().await.unwrap();

        // Select only specific columns
        let selected = df.select(vec![col("name"), col("value")]).unwrap();

        // Convert back to Cylon Table
        let result = Table::from_datafusion(ctx, selected).await.unwrap();

        // Should have only 2 columns now
        assert_eq!(result.columns(), 2);
        assert_eq!(result.rows(), 5);
        assert_eq!(result.column_names(), vec!["name", "value"]);
    }

    #[tokio::test]
    async fn test_datafusion_aggregate() {
        use datafusion::prelude::*;
        use datafusion::functions_aggregate::expr_fn::{count, sum};

        let (ctx, table) = create_test_table();

        // Convert to DataFusion DataFrame
        let df = table.to_datafusion().await.unwrap();

        // Compute aggregate
        let aggregated = df.aggregate(
            vec![],  // no grouping
            vec![
                count(col("id")).alias("count"),
                sum(col("value")).alias("total"),
            ]
        ).unwrap();

        // Convert back to Cylon Table
        let result = Table::from_datafusion(ctx, aggregated).await.unwrap();

        // Should have 1 row with aggregated values
        assert_eq!(result.rows(), 1);
        assert_eq!(result.columns(), 2);
    }

    #[tokio::test]
    async fn test_empty_table_conversion() {
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

        // Convert to DataFusion and back
        let df = table.to_datafusion().await.unwrap();
        let result = Table::from_datafusion(ctx, df).await.unwrap();

        assert_eq!(result.rows(), 0);
        assert_eq!(result.columns(), 1);
    }
}

#[cfg(not(feature = "datafusion"))]
mod datafusion_disabled {
    #[test]
    fn datafusion_feature_not_enabled() {
        println!("DataFusion tests skipped - feature not enabled");
    }
}
