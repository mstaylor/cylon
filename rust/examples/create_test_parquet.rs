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

//! Utility to create test parquet files for examples

use std::sync::Arc;
use arrow::array::{Int32Array, StringArray};
use arrow::datatypes::{Schema, Field, DataType};
use arrow::record_batch::RecordBatch;
use cylon::ctx::CylonContext;
use cylon::table::Table;

fn main() {
    let ctx = Arc::new(CylonContext::new(false));

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
    ]));

    // Create first test file
    let batch1 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])),
            Arc::new(StringArray::from(vec!["alice", "bob", "charlie", "david", "eve"])),
        ],
    ).unwrap();

    let table1 = Table::from_record_batch(ctx.clone(), batch1).unwrap();
    table1.to_parquet_default("/tmp/test1.parquet").unwrap();
    println!("Created /tmp/test1.parquet with {} rows", table1.rows());

    // Create second test file with overlapping IDs
    let batch2 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![3, 4, 5, 6, 7])),
            Arc::new(StringArray::from(vec!["charlie2", "david2", "eve2", "frank", "grace"])),
        ],
    ).unwrap();

    let table2 = Table::from_record_batch(ctx.clone(), batch2).unwrap();
    table2.to_parquet_default("/tmp/test2.parquet").unwrap();
    println!("Created /tmp/test2.parquet with {} rows", table2.rows());

    println!("\nTest parquet files created successfully!");
    println!("You can now run:");
    println!("  mpirun -np 2 ./target/debug/examples/parquet_join_example /tmp/test1.parquet /tmp/test2.parquet");
    println!("  mpirun -np 2 ./target/debug/examples/parquet_union_example /tmp/test1.parquet /tmp/test2.parquet");
}
