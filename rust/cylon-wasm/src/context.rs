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

//! WASM bindings for the Arrow-native ContextTable.
//!
//! Exposes `WasmContextTable` to JavaScript/Node.js via wasm-bindgen.
//! Metadata is passed as JSON strings for JS compatibility.

use wasm_bindgen::prelude::*;

use cylon::context::{ContextMetadata, ContextTable};

/// WASM-exported ContextTable for embedding storage and SIMD similarity search.
///
/// # JavaScript Usage
/// ```javascript
/// const table = new WasmContextTable(1024);
/// table.put("ctx-1", new Float32Array([...]), JSON.stringify({
///     workflow_id: "wf-1", response: "Hello", model_id: "claude"
/// }));
/// const results = JSON.parse(table.search(new Float32Array([...]), 0.85, 5));
/// ```
#[wasm_bindgen]
pub struct WasmContextTable {
    inner: ContextTable,
}

#[wasm_bindgen]
impl WasmContextTable {
    /// Create a new empty ContextTable with the given embedding dimension.
    #[wasm_bindgen(constructor)]
    pub fn new(embedding_dim: usize) -> Result<WasmContextTable, JsValue> {
        let inner = ContextTable::new(embedding_dim)
            .map_err(|e| JsValue::from_str(&format!("{}", e)))?;
        Ok(Self { inner })
    }

    /// Insert or update a context entry.
    /// `metadata_json` is a JSON string with optional fields:
    /// workflow_id, response, model_id, input_tokens, output_tokens, cost_usd
    pub fn put(
        &mut self,
        context_id: &str,
        embedding: &[f32],
        metadata_json: &str,
    ) -> Result<(), JsValue> {
        let metadata = parse_metadata(metadata_json)?;
        self.inner
            .put(context_id, embedding, metadata)
            .map_err(|e| JsValue::from_str(&format!("{}", e)))
    }

    /// Retrieve a single row by context_id.
    /// Returns JSON string of the row, or null if not found.
    pub fn get(&self, context_id: &str) -> Result<JsValue, JsValue> {
        match self.inner.get(context_id) {
            Some(_batch) => {
                // Return a simple JSON with the context_id to confirm existence
                // Full row data would require Arrow IPC or column extraction
                Ok(JsValue::from_str(context_id))
            }
            None => Ok(JsValue::NULL),
        }
    }

    /// Mark a row as deleted.
    pub fn remove(&mut self, context_id: &str) -> Result<(), JsValue> {
        self.inner
            .remove(context_id)
            .map_err(|e| JsValue::from_str(&format!("{}", e)))
    }

    /// SIMD cosine similarity search.
    /// Returns JSON string array of {index, similarity} sorted by descending similarity.
    pub fn search(
        &self,
        query: &[f32],
        threshold: f32,
        top_k: usize,
    ) -> Result<String, JsValue> {
        let results = self.inner.search(query, threshold, top_k, None);
        let json: Vec<serde_json::Value> = results
            .iter()
            .map(|r| serde_json::json!({"index": r.index, "similarity": r.similarity}))
            .collect();
        serde_json::to_string(&json)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// SIMD cosine similarity search filtered by workflow_id.
    pub fn search_workflow(
        &self,
        query: &[f32],
        workflow_id: &str,
        threshold: f32,
        top_k: usize,
    ) -> Result<String, JsValue> {
        let results = self.inner.search(query, threshold, top_k, Some(workflow_id));
        let json: Vec<serde_json::Value> = results
            .iter()
            .map(|r| serde_json::json!({"index": r.index, "similarity": r.similarity}))
            .collect();
        serde_json::to_string(&json)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Serialize to Arrow IPC bytes.
    pub fn to_ipc(&self) -> Result<Vec<u8>, JsValue> {
        self.inner
            .to_ipc()
            .map_err(|e| JsValue::from_str(&format!("{}", e)))
    }

    /// Deserialize from Arrow IPC bytes.
    pub fn from_ipc(data: &[u8]) -> Result<WasmContextTable, JsValue> {
        let inner = ContextTable::from_ipc(data)
            .map_err(|e| JsValue::from_str(&format!("{}", e)))?;
        Ok(Self { inner })
    }

    /// Remove tombstoned rows.
    pub fn compact(&mut self) -> Result<(), JsValue> {
        self.inner
            .compact()
            .map_err(|e| JsValue::from_str(&format!("{}", e)))
    }

    /// Number of active (non-deleted) rows.
    pub fn size(&self) -> usize {
        self.inner.len()
    }

    /// Embedding dimension.
    pub fn embedding_dim(&self) -> usize {
        self.inner.embedding_dim()
    }
}

fn parse_metadata(json: &str) -> Result<ContextMetadata, JsValue> {
    let v: serde_json::Value = serde_json::from_str(json)
        .map_err(|e| JsValue::from_str(&format!("Invalid metadata JSON: {}", e)))?;
    Ok(ContextMetadata {
        workflow_id: v["workflow_id"].as_str().unwrap_or("").to_string(),
        response: v["response"].as_str().unwrap_or("").to_string(),
        model_id: v["model_id"].as_str().unwrap_or("").to_string(),
        input_tokens: v["input_tokens"].as_i64().unwrap_or(0),
        output_tokens: v["output_tokens"].as_i64().unwrap_or(0),
        cost_usd: v["cost_usd"].as_f64().unwrap_or(0.0),
    })
}