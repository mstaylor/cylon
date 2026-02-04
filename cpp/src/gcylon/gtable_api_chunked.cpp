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

#include <gcylon/gtable_api.hpp>
#include <gcylon/staging/memory_utils.hpp>
#include <gcylon/staging/pinned_allocator.hpp>
#include <gcylon/net/cudf_net_ops.hpp>

#include <cylon/util/macros.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>

#include <algorithm>
#include <vector>

namespace gcylon {

namespace {

/**
 * Split a table into chunks by row count.
 *
 * @param tv Table view to split
 * @param rows_per_chunk Maximum rows per chunk
 * @return Vector of table views, each representing a chunk
 */
std::vector<cudf::table_view> split_table(
    const cudf::table_view& tv,
    size_t rows_per_chunk
) {
    std::vector<cudf::table_view> chunks;
    int64_t total_rows = tv.num_rows();

    if (total_rows <= 0) {
        return chunks;
    }

    // If the table is smaller than chunk size, return it as a single chunk
    if (static_cast<size_t>(total_rows) <= rows_per_chunk) {
        chunks.push_back(tv);
        return chunks;
    }

    // Build split indices
    std::vector<cudf::size_type> split_indices;
    for (int64_t start = static_cast<int64_t>(rows_per_chunk);
         start < total_rows;
         start += static_cast<int64_t>(rows_per_chunk)) {
        split_indices.push_back(static_cast<cudf::size_type>(start));
    }

    // Use cudf::split to create the chunks
    auto split_tables = cudf::split(tv, split_indices);
    for (auto& chunk : split_tables) {
        if (chunk.num_rows() > 0) {
            chunks.push_back(chunk);
        }
    }

    return chunks;
}

/**
 * Concatenate multiple tables into one.
 *
 * @param tables Vector of unique_ptr to tables
 * @return Single concatenated table
 */
std::unique_ptr<cudf::table> concatenate_tables(
    std::vector<std::unique_ptr<cudf::table>>& tables
) {
    if (tables.empty()) {
        return nullptr;
    }

    if (tables.size() == 1) {
        return std::move(tables[0]);
    }

    std::vector<cudf::table_view> views;
    views.reserve(tables.size());
    for (const auto& t : tables) {
        if (t && t->num_rows() > 0) {
            views.push_back(t->view());
        }
    }

    if (views.empty()) {
        return std::move(tables[0]);
    }

    return cudf::concatenate(views);
}

} // anonymous namespace

cylon::Status ChunkedShuffle(
    const cudf::table_view &input_tv,
    const std::vector<int> &columns_to_hash,
    const std::shared_ptr<cylon::CylonContext> &ctx,
    std::unique_ptr<cudf::table> &table_out,
    const GcylonConfig &config
) {
    if (input_tv.num_rows() == 0) {
        table_out = cudf::empty_like(input_tv);
        return cylon::Status::OK();
    }

    int world_size = ctx->GetWorldSize();

    // Calculate available memory
    size_t available = get_available_memory(config.gpu_memory_limit, config.gpu_memory_fraction);

    // Calculate rows per chunk
    size_t rows_per_chunk;
    if (config.chunk_size_bytes > 0) {
        size_t table_size = estimate_table_size(input_tv);
        size_t row_size = table_size / static_cast<size_t>(input_tv.num_rows());
        rows_per_chunk = row_size > 0 ? config.chunk_size_bytes / row_size : config.min_chunk_rows;
    } else {
        rows_per_chunk = calculate_chunk_rows(input_tv, world_size, available, config.min_chunk_rows);
    }

    // Clamp rows_per_chunk
    rows_per_chunk = std::max(rows_per_chunk, config.min_chunk_rows);

    // If table fits in one chunk, use fast path
    if (static_cast<size_t>(input_tv.num_rows()) <= rows_per_chunk) {
        return Shuffle(input_tv, columns_to_hash, ctx, table_out);
    }

    // Split into chunks
    auto chunks = split_table(input_tv, rows_per_chunk);

    // Process each chunk
    std::vector<std::unique_ptr<cudf::table>> results;
    results.reserve(chunks.size());

    for (size_t i = 0; i < chunks.size(); i++) {
        // Partition this chunk
        auto [partitioned, offsets] = cudf::hash_partition(
            chunks[i], columns_to_hash, world_size
        );
        offsets.push_back(chunks[i].num_rows());

        // AllToAll exchange
        std::unique_ptr<cudf::table> chunk_result;
        RETURN_CYLON_STATUS_IF_FAILED(
            gcylon::net::AllToAll(partitioned->view(), offsets, ctx, chunk_result)
        );

        // Store result
        results.push_back(std::move(chunk_result));

        // partitioned goes out of scope, freeing GPU memory
    }

    // Concatenate all results
    table_out = concatenate_tables(results);

    return cylon::Status::OK();
}

cylon::Status ChunkedAllGather(
    const cudf::table_view &input_tv,
    const std::shared_ptr<cylon::CylonContext> &ctx,
    std::unique_ptr<cudf::table> &table_out,
    const GcylonConfig &config
) {
    if (input_tv.num_rows() == 0) {
        // Still need to gather empty tables from all workers
        return AllGather(input_tv, ctx, table_out);
    }

    int world_size = ctx->GetWorldSize();

    // Calculate available memory
    size_t available = get_available_memory(config.gpu_memory_limit, config.gpu_memory_fraction);

    // For allgather, output is world_size times larger
    size_t output_size = estimate_table_size(input_tv) * static_cast<size_t>(world_size);

    // If output fits in memory, use fast path
    if (output_size < available) {
        return AllGather(input_tv, ctx, table_out);
    }

    // Calculate rows per chunk
    // For allgather, each chunk becomes world_size chunks in output
    size_t table_size = estimate_table_size(input_tv);
    size_t row_size = table_size / static_cast<size_t>(input_tv.num_rows());
    if (row_size == 0) row_size = 1;

    size_t rows_per_chunk = available / (row_size * static_cast<size_t>(world_size + 2));
    rows_per_chunk = std::max(rows_per_chunk, config.min_chunk_rows);

    auto chunks = split_table(input_tv, rows_per_chunk);

    std::vector<std::unique_ptr<cudf::table>> results;
    results.reserve(chunks.size());

    for (size_t i = 0; i < chunks.size(); i++) {
        std::vector<std::unique_ptr<cudf::table>> gathered;

        RETURN_CYLON_STATUS_IF_FAILED(
            gcylon::net::AllGather(chunks[i], ctx, gathered)
        );

        // Concatenate this round's gathered chunks
        std::vector<cudf::table_view> views;
        views.reserve(gathered.size());
        for (const auto& t : gathered) {
            if (t && t->num_rows() > 0) {
                views.push_back(t->view());
            }
        }

        if (!views.empty()) {
            auto merged = cudf::concatenate(views);
            results.push_back(std::move(merged));
        }

        // gathered goes out of scope, freeing memory
    }

    // Final concatenation
    table_out = concatenate_tables(results);

    return cylon::Status::OK();
}

cylon::Status SmartShuffle(
    const cudf::table_view &input_tv,
    const std::vector<int> &columns_to_hash,
    const std::shared_ptr<cylon::CylonContext> &ctx,
    std::unique_ptr<cudf::table> &table_out,
    const GcylonConfig &config
) {
    if (input_tv.num_rows() == 0) {
        table_out = cudf::empty_like(input_tv);
        return cylon::Status::OK();
    }

    size_t estimated = estimate_shuffle_memory(input_tv, ctx->GetWorldSize());

    // Use 50% as threshold for "comfortable" memory
    if (fits_in_gpu_memory(estimated, 0.5f)) {
        // Plenty of room - use fast path
        return Shuffle(input_tv, columns_to_hash, ctx, table_out);
    } else {
        // Memory pressure - use chunked path
        return ChunkedShuffle(input_tv, columns_to_hash, ctx, table_out, config);
    }
}

cylon::Status SmartAllGather(
    const cudf::table_view &input_tv,
    const std::shared_ptr<cylon::CylonContext> &ctx,
    std::unique_ptr<cudf::table> &table_out,
    const GcylonConfig &config
) {
    if (input_tv.num_rows() == 0) {
        return AllGather(input_tv, ctx, table_out);
    }

    size_t output_size = estimate_allgather_memory(input_tv, ctx->GetWorldSize());

    // Use 50% as threshold for "comfortable" memory
    if (fits_in_gpu_memory(output_size, 0.5f)) {
        return AllGather(input_tv, ctx, table_out);
    } else {
        return ChunkedAllGather(input_tv, ctx, table_out, config);
    }
}

} // namespace gcylon
