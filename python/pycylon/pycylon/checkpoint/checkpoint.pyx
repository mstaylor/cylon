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

from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libc.stdint cimport uint64_t

from pycylon.common.status cimport CStatus
from pycylon.data.table cimport CTable
from pycylon.ctx.context cimport CCylonContext, CylonContext
from pycylon.checkpoint.checkpoint cimport (
    CCheckpointConfig,
    CCheckpointManager,
    COperationType,
)
from pycylon.api.lib cimport pycylon_unwrap_context, pycylon_unwrap_table, pycylon_wrap_table


cpdef enum OperationType:
    JOIN = COperationType._Join
    FILTER = COperationType._Filter
    SORT = COperationType._Sort
    GROUPBY = COperationType._GroupBy
    SETOP = COperationType._SetOp
    SHUFFLE = COperationType._Shuffle
    OTHER = COperationType._Other


cdef class CheckpointConfig:
    """Configuration for the checkpoint system."""

    def __cinit__(self, str job_id="default", str storage_path="/tmp/cylon_checkpoints",
                  uint64_t operation_threshold=100,
                  uint64_t bytes_threshold=104857600,
                  int max_checkpoints=10, int min_retain=3):
        self.c_config = CCheckpointConfig(job_id.encode())
        self.c_config.storage_path = storage_path.encode()
        self.c_config.trigger.operation_threshold = operation_threshold
        self.c_config.trigger.bytes_threshold = bytes_threshold
        self.c_config.retention.max_checkpoints = max_checkpoints
        self.c_config.retention.min_retain = min_retain

    @property
    def job_id(self) -> str:
        return self.c_config.job_id.decode()

    @property
    def storage_path(self) -> str:
        return self.c_config.storage_path.decode()


cdef class CheckpointManager:
    """Orchestrates checkpoint lifecycle for Cylon tables.

    Manages registered tables, coordinates with other workers,
    serializes data, and writes to storage. All operations are synchronous.

    Example:
        >>> ctx = CylonContext(config=config, distributed=True)
        >>> ckpt_config = CheckpointConfig(job_id="my_job", storage_path="/tmp/ckpts")
        >>> manager = CheckpointManager(ctx, ckpt_config, distributed=True)
        >>> manager.register_table("orders", orders_table)
        >>> checkpoint_id = manager.checkpoint()
        >>> tables = manager.restore()
    """

    def __cinit__(self, CylonContext ctx not None, CheckpointConfig config not None,
                  bint distributed=False):
        cdef CStatus status
        cdef shared_ptr[CCylonContext] ctx_ptr = pycylon_unwrap_context(ctx)
        cdef CCheckpointConfig c_cfg = config.c_config

        if distributed:
            status = CCheckpointManager.MakeDistributed(
                ctx_ptr, c_cfg, &self.manager_ptr)
        else:
            status = CCheckpointManager.MakeLocal(
                ctx_ptr, c_cfg, &self.manager_ptr)

        if not status.is_ok():
            raise Exception(
                f"Failed to create CheckpointManager: {status.get_msg().decode()}")

    def register_table(self, str name not None, table not None):
        """Register a table for checkpointing.

        Args:
            name: Name to identify this table in checkpoints.
            table: The Cylon Table to checkpoint.
        """
        cdef shared_ptr[CTable] c_table = pycylon_unwrap_table(table)
        self.manager_ptr.get().RegisterTable(name.encode(), c_table)

    def update_table(self, str name not None, table not None):
        """Update a previously registered table.

        Args:
            name: Name of the registered table.
            table: The updated Cylon Table.
        """
        cdef shared_ptr[CTable] c_table = pycylon_unwrap_table(table)
        self.manager_ptr.get().UpdateTable(name.encode(), c_table)

    def unregister_table(self, str name not None):
        """Remove a table from checkpointing.

        Args:
            name: Name of the table to remove.
        """
        self.manager_ptr.get().UnregisterTable(name.encode())

    def record_operation(self, op_type: OperationType = OperationType.OTHER,
                         uint64_t bytes_processed=0):
        """Record an operation for the trigger.

        Args:
            op_type: Type of operation performed.
            bytes_processed: Number of bytes processed.
        """
        self.manager_ptr.get().RecordOperation(
            <COperationType> op_type, bytes_processed)

    def should_checkpoint(self) -> bool:
        """Check if a checkpoint should be triggered."""
        return self.manager_ptr.get().ShouldCheckpoint()

    def checkpoint(self) -> int:
        """Perform a checkpoint of all registered tables.

        Returns:
            The checkpoint ID (0 if skipped by coordinator).

        Raises:
            Exception: If the checkpoint fails.
        """
        cdef uint64_t checkpoint_id = 0
        cdef CStatus status = self.manager_ptr.get().Checkpoint(&checkpoint_id)
        if not status.is_ok():
            raise Exception(
                f"Checkpoint failed: {status.get_msg().decode()}")
        return checkpoint_id

    def restore(self) -> dict:
        """Restore from the latest available checkpoint.

        Returns:
            Dictionary mapping table names to restored Cylon Tables.

        Raises:
            Exception: If restore fails.
        """
        cdef unordered_map[string, shared_ptr[CTable]] c_tables
        cdef CStatus status = self.manager_ptr.get().Restore(&c_tables)
        if not status.is_ok():
            raise Exception(
                f"Restore failed: {status.get_msg().decode()}")

        result = {}
        for pair in c_tables:
            result[pair.first.decode()] = pycylon_wrap_table(pair.second)
        return result

    def restore_from(self, uint64_t checkpoint_id) -> dict:
        """Restore from a specific checkpoint.

        Args:
            checkpoint_id: The checkpoint ID to restore from.

        Returns:
            Dictionary mapping table names to restored Cylon Tables.

        Raises:
            Exception: If restore fails.
        """
        cdef unordered_map[string, shared_ptr[CTable]] c_tables
        cdef CStatus status = self.manager_ptr.get().RestoreFrom(
            checkpoint_id, &c_tables)
        if not status.is_ok():
            raise Exception(
                f"Restore failed: {status.get_msg().decode()}")

        result = {}
        for pair in c_tables:
            result[pair.first.decode()] = pycylon_wrap_table(pair.second)
        return result

    def prune(self):
        """Prune old checkpoints according to retention policy.

        Raises:
            Exception: If pruning fails.
        """
        cdef CStatus status = self.manager_ptr.get().Prune()
        if not status.is_ok():
            raise Exception(
                f"Prune failed: {status.get_msg().decode()}")

    @property
    def next_checkpoint_id(self) -> int:
        """Get the next checkpoint ID."""
        return self.manager_ptr.get().NextCheckpointId()