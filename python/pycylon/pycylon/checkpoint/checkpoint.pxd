##
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##

from libcpp cimport bool
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libc.stdint cimport uint64_t

from pycylon.common.status cimport CStatus
from pycylon.data.table cimport CTable
from pycylon.ctx.context cimport CCylonContext


cdef extern from "../../../../cpp/src/cylon/checkpoint/checkpoint_types.hpp" namespace "cylon::checkpoint":
    cdef enum COperationType "cylon::checkpoint::OperationType":
        _Join "cylon::checkpoint::OperationType::Join"
        _Filter "cylon::checkpoint::OperationType::Filter"
        _Sort "cylon::checkpoint::OperationType::Sort"
        _GroupBy "cylon::checkpoint::OperationType::GroupBy"
        _SetOp "cylon::checkpoint::OperationType::SetOp"
        _Shuffle "cylon::checkpoint::OperationType::Shuffle"
        _Other "cylon::checkpoint::OperationType::Other"

    cdef enum CCheckpointStatus "cylon::checkpoint::CheckpointStatus":
        _InProgress "cylon::checkpoint::CheckpointStatus::InProgress"
        _Committed "cylon::checkpoint::CheckpointStatus::Committed"
        _Failed "cylon::checkpoint::CheckpointStatus::Failed"
        _Aborted "cylon::checkpoint::CheckpointStatus::Aborted"


cdef extern from "../../../../cpp/src/cylon/checkpoint/checkpoint_config.hpp" namespace "cylon::checkpoint":
    cdef cppclass CTriggerConfig "cylon::checkpoint::TriggerConfig":
        uint64_t operation_threshold
        uint64_t bytes_threshold

    cdef cppclass CPrunePolicy "cylon::checkpoint::PrunePolicy":
        int max_checkpoints
        int max_age_seconds
        int min_retain

    cdef cppclass CCheckpointConfig "cylon::checkpoint::CheckpointConfig":
        CCheckpointConfig()
        CCheckpointConfig(string job_id)
        string job_id
        string storage_path
        CTriggerConfig trigger
        CPrunePolicy retention


cdef extern from "../../../../cpp/src/cylon/checkpoint/checkpoint_manager.hpp" namespace "cylon::checkpoint":
    cdef cppclass CCheckpointManager "cylon::checkpoint::CheckpointManager":
        void RegisterTable(const string &name, shared_ptr[CTable] table)
        void UpdateTable(const string &name, shared_ptr[CTable] table)
        void UnregisterTable(const string &name)
        void RecordOperation(COperationType op_type, uint64_t bytes_processed)
        bool ShouldCheckpoint()
        CStatus Checkpoint(uint64_t *checkpoint_id)
        CStatus Restore(unordered_map[string, shared_ptr[CTable]] *tables)
        CStatus RestoreFrom(uint64_t checkpoint_id,
                            unordered_map[string, shared_ptr[CTable]] *tables)
        CStatus Prune()
        uint64_t NextCheckpointId()

        @staticmethod
        CStatus MakeLocal(const shared_ptr[CCylonContext] &ctx,
                          const CCheckpointConfig &config,
                          unique_ptr[CCheckpointManager] *manager)

        @staticmethod
        CStatus MakeDistributed(const shared_ptr[CCylonContext] &ctx,
                                const CCheckpointConfig &config,
                                unique_ptr[CCheckpointManager] *manager)


cdef class CheckpointConfig:
    cdef:
        CCheckpointConfig c_config

cdef class CheckpointManager:
    cdef:
        unique_ptr[CCheckpointManager] manager_ptr