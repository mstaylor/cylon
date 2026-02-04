/**
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

/**
 * AWS Lambda handler for Cylon WASM operations (Node.js runtime).
 *
 * Supports join, groupby, filter, and aggregate operations via WASM.
 *
 * Environment variables:
 *   CYLON_WASM_PATH: Path to cylon_wasm_bg.wasm file (optional)
 *
 * Example event:
 *   {
 *     "operation": "join",
 *     "left": {"columns": [...], "data": [...]},
 *     "right": {"columns": [...], "data": [...]},
 *     "config": {"join_type": "inner", "left_on": [0], "right_on": [0]}
 *   }
 */

import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

// Get __dirname equivalent for ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Lazy-loaded WASM module
let wasmModule = null;

/**
 * Initialize the WASM module.
 */
async function initWasm() {
    if (wasmModule) {
        return wasmModule;
    }

    // Determine WASM path
    const wasmPath = process.env.CYLON_WASM_PATH ||
        join(__dirname, '..', '..', '..', '..', 'rust', 'cylon-wasm', 'pkg', 'cylon_wasm_bg.wasm');

    // Load the JS bindings
    const bindingsPath = process.env.CYLON_WASM_BINDINGS ||
        join(__dirname, '..', '..', '..', '..', 'rust', 'cylon-wasm', 'pkg', 'cylon_wasm.js');

    // Import the bindings module
    const bindings = await import(bindingsPath);

    // Read WASM binary
    const wasmBytes = readFileSync(wasmPath);

    // Initialize with the WASM bytes (new API style)
    await bindings.default({ module_or_path: wasmBytes });

    // Call init to set up panic hook
    if (bindings.init) {
        bindings.init();
    }

    wasmModule = bindings;
    return wasmModule;
}

/**
 * Lambda handler for WASM DataFrame operations.
 *
 * @param {Object} event - Lambda event
 * @param {Object} context - Lambda context
 * @returns {Object} Response with statusCode and body
 */
export async function handler(event, context) {
    try {
        const operation = event.operation;
        if (!operation) {
            return {
                statusCode: 400,
                body: JSON.stringify({ error: "Missing 'operation' field" })
            };
        }

        const wasm = await initWasm();
        let result;

        switch (operation) {
            case 'join':
                result = handleJoin(wasm, event);
                break;
            case 'groupby':
                result = handleGroupBy(wasm, event);
                break;
            case 'filter':
                result = handleFilter(wasm, event);
                break;
            case 'aggregate':
                result = handleAggregate(wasm, event);
                break;
            case 'table_info':
                result = handleTableInfo(wasm, event);
                break;
            case 'pipeline':
                result = handlePipeline(wasm, event);
                break;
            default:
                return {
                    statusCode: 400,
                    body: JSON.stringify({ error: `Unknown operation: ${operation}` })
                };
        }

        return {
            statusCode: 200,
            body: typeof result === 'string' ? result : JSON.stringify(result)
        };

    } catch (error) {
        console.error('Error processing request:', error);
        return {
            statusCode: 500,
            body: JSON.stringify({ error: error.message })
        };
    }
}

/**
 * Handle join operation.
 */
function handleJoin(wasm, event) {
    const leftData = event.left;
    const rightData = event.right;
    const config = event.config || {};

    if (!leftData || !rightData) {
        throw new Error("Join requires 'left' and 'right' tables");
    }

    const result = wasm.join_tables(
        JSON.stringify(leftData),
        JSON.stringify(rightData),
        JSON.stringify({
            join_type: config.join_type || 'inner',
            left_on: config.left_on || [0],
            right_on: config.right_on || [0],
            left_suffix: config.left_suffix || '_l',
            right_suffix: config.right_suffix || '_r'
        })
    );

    return result;
}

/**
 * Handle groupby operation.
 */
function handleGroupBy(wasm, event) {
    const tableData = event.table;
    const config = event.config || {};

    if (!tableData) {
        throw new Error("GroupBy requires 'table'");
    }

    const result = wasm.groupby_table(
        JSON.stringify(tableData),
        JSON.stringify({
            keys: config.keys || [0],
            aggregations: config.aggregations || []
        })
    );

    return result;
}

/**
 * Handle filter operation.
 */
function handleFilter(wasm, event) {
    const tableData = event.table;
    const config = event.config || {};

    if (!tableData) {
        throw new Error("Filter requires 'table'");
    }

    const result = wasm.filter_table(
        JSON.stringify(tableData),
        JSON.stringify({
            predicates: config.predicates || [],
            logic: config.logic || 'and'
        })
    );

    return result;
}

/**
 * Handle aggregate operation.
 */
function handleAggregate(wasm, event) {
    const tableData = event.table;
    const column = event.column;
    const op = event.op;

    if (!tableData) {
        throw new Error("Aggregate requires 'table'");
    }
    if (column === undefined || column === null) {
        throw new Error("Aggregate requires 'column'");
    }
    if (!op) {
        throw new Error("Aggregate requires 'op'");
    }

    const result = wasm.aggregate(
        JSON.stringify(tableData),
        column,
        op
    );

    return { result };
}

/**
 * Get table info.
 */
function handleTableInfo(wasm, event) {
    const tableData = event.table;

    if (!tableData) {
        throw new Error("table_info requires 'table'");
    }

    return wasm.table_info(JSON.stringify(tableData));
}

/**
 * Execute a pipeline of operations.
 */
function handlePipeline(wasm, event) {
    const tableData = event.table;
    const pipeline = event.pipeline;

    if (!tableData) {
        throw new Error("Pipeline requires 'table'");
    }
    if (!pipeline) {
        throw new Error("Pipeline requires 'pipeline'");
    }

    return wasm.execute_pipeline(
        JSON.stringify(tableData),
        JSON.stringify(pipeline)
    );
}

// For local testing
const isMain = import.meta.url === `file://${process.argv[1]}`;
if (isMain) {
    const testEvent = {
        operation: 'join',
        left: {
            columns: ['id', 'name'],
            data: [
                { type: 'Int32', data: [1, 2, 3] },
                { type: 'String', data: ['Alice', 'Bob', 'Carol'] }
            ]
        },
        right: {
            columns: ['id', 'dept'],
            data: [
                { type: 'Int32', data: [1, 2, 4] },
                { type: 'String', data: ['Eng', 'Sales', 'Marketing'] }
            ]
        },
        config: {
            join_type: 'inner',
            left_on: [0],
            right_on: [0]
        }
    };

    handler(testEvent, {}).then(result => {
        console.log(JSON.stringify(result, null, 2));
    });
}
