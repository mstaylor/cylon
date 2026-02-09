/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "common/test_header.hpp"
#include "gcylon/test_gutils.hpp"
#include <gcylon/utils/util.hpp>
#include <gcylon/gcylon_config.hpp>

using namespace cylon;
using namespace gcylon;

TEST_CASE("chunked shuffle operations", "[chunked_shuffle]") {

  SECTION("SmartShuffle small table uses direct path and produces correct results") {
    std::string input_filename = "../../data/input/cities_a_" + std::to_string(RANK) + ".csv";
    std::string expected_filename = "../../data/output/shuffle_int_cities_a_" + std::to_string(RANK) + ".csv";

    std::vector<std::string> column_names{"city", "state_id", "population"};
    cudf::io::table_with_metadata input_table = gcylon::test::readCSV(input_filename, column_names);
    auto input_tv = input_table.tbl->view();

    // SmartShuffle on small data — should route to direct path
    std::vector<int> columns_to_hash = {2}; // population
    std::unique_ptr<cudf::table> smart_result;
    auto status = SmartShuffle(input_tv, columns_to_hash, ctx, smart_result);
    REQUIRE(status.is_ok());

    // Compare against expected output
    cudf::io::table_with_metadata expected_table = gcylon::test::readCSV(expected_filename, column_names);
    auto smart_tv = smart_result->view();
    auto expected_tv = expected_table.tbl->view();
    REQUIRE(table_equal_with_sorting(smart_tv, expected_tv));
  }

  SECTION("ChunkedShuffle forced via low memory fraction produces correct results") {
    std::string input_filename = "../../data/input/cities_a_" + std::to_string(RANK) + ".csv";

    std::vector<std::string> column_names{"city", "state_id", "population"};
    cudf::io::table_with_metadata input_table = gcylon::test::readCSV(input_filename, column_names);
    auto input_tv = input_table.tbl->view();

    std::vector<int> columns_to_hash = {2}; // population

    // Direct shuffle for reference result
    std::unique_ptr<cudf::table> direct_result;
    auto status = Shuffle(input_tv, columns_to_hash, ctx, direct_result);
    REQUIRE(status.is_ok());

    // Force chunking with very low memory fraction
    GcylonConfig config;
    config.gpu_memory_fraction = 0.1f;
    config.min_chunk_rows = 2;

    std::unique_ptr<cudf::table> chunked_result;
    status = ChunkedShuffle(input_tv, columns_to_hash, ctx, chunked_result, config);
    REQUIRE(status.is_ok());

    // Chunked result must match direct result
    auto direct_tv = direct_result->view();
    auto chunked_tv = chunked_result->view();
    REQUIRE(table_equal_with_sorting(direct_tv, chunked_tv));
  }

  SECTION("ChunkedShuffle empty table produces empty output") {
    std::string input_filename = "../../data/input/cities_a_" + std::to_string(RANK) + ".csv";

    std::vector<std::string> column_names{"city", "state_id", "population"};
    cudf::io::table_with_metadata input_table = gcylon::test::readCSV(input_filename, column_names);

    // Create an empty slice of the input to preserve schema
    auto empty_tv = cudf::slice(input_table.tbl->view(), {0, 0})[0];
    REQUIRE(empty_tv.num_rows() == 0);

    std::vector<int> columns_to_hash = {2};
    GcylonConfig config;
    config.gpu_memory_fraction = 0.1f;

    std::unique_ptr<cudf::table> result;
    auto status = ChunkedShuffle(empty_tv, columns_to_hash, ctx, result, config);
    REQUIRE(status.is_ok());
    REQUIRE(result->num_rows() == 0);
  }
}

TEST_CASE("chunked allgather operations", "[chunked_allgather]") {

  SECTION("SmartAllGather small table uses direct path and produces correct results") {
    std::string input_file_base = "../../data/mpiops/sales_nulls_nunascii_";

    std::vector<std::string> column_names{"Country", "Item Type", "Order Date", "Order ID", "Units Sold", "Unit Price"};
    std::vector<std::string> date_columns{"Order Date"};

    auto tables = gcylon::test::readTables(input_file_base, column_names, date_columns);
    REQUIRE((tables.size() == WORLD_SZ));

    auto input_tv = tables[RANK]->view();

    // Direct AllGather for reference result
    std::unique_ptr<cudf::table> direct_result;
    auto status = AllGather(input_tv, ctx, direct_result);
    REQUIRE(status.is_ok());

    // SmartAllGather on small data — should route to direct path
    std::unique_ptr<cudf::table> smart_result;
    status = SmartAllGather(input_tv, ctx, smart_result);
    REQUIRE(status.is_ok());

    // Results must match
    REQUIRE(table_equal(direct_result->view(), smart_result->view()));
  }

  SECTION("ChunkedAllGather forced via low memory fraction produces correct results") {
    std::string input_file_base = "../../data/mpiops/sales_nulls_nunascii_";

    std::vector<std::string> column_names{"Country", "Item Type", "Order Date", "Order ID", "Units Sold", "Unit Price"};
    std::vector<std::string> date_columns{"Order Date"};

    auto tables = gcylon::test::readTables(input_file_base, column_names, date_columns);
    REQUIRE((tables.size() == WORLD_SZ));

    auto input_tv = tables[RANK]->view();

    // Direct AllGather for reference result
    std::unique_ptr<cudf::table> direct_result;
    auto status = AllGather(input_tv, ctx, direct_result);
    REQUIRE(status.is_ok());

    // Force chunking with very low memory fraction
    GcylonConfig config;
    config.gpu_memory_fraction = 0.1f;
    config.min_chunk_rows = 2;

    std::unique_ptr<cudf::table> chunked_result;
    status = ChunkedAllGather(input_tv, ctx, chunked_result, config);
    REQUIRE(status.is_ok());

    // Chunked result must match direct result
    REQUIRE(table_equal(direct_result->view(), chunked_result->view()));
  }
}