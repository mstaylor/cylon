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

from libcpp.memory cimport shared_ptr
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector

import cudf
from cudf._lib.column import Column
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.table cimport Table as plc_Table

from pycylon.ctx.context cimport CCylonContext
from pycylon.api.lib cimport pycylon_unwrap_context
from pygcylon.net.c_comms cimport Repartition, Gather, Broadcast, AllGather


cdef table_view _dataframe_to_table_view(df, bint ignore_index):
    """Convert cudf DataFrame to libcudf table_view."""
    cdef list plc_columns = []
    cdef plc_Table plc_table
    # Convert data columns
    for col in df._columns:
        plc_columns.append(col.to_pylibcudf(mode="read"))
    # Include index columns unless ignored
    if not ignore_index and df.index is not None:
        for idx_col in df.index._columns:
            plc_columns.insert(0, idx_col.to_pylibcudf(mode="read"))
    plc_table = plc_Table(plc_columns)
    return plc_table.view()


cdef _table_to_dataframe(unique_ptr[table]& c_table_ptr, column_names, index_names):
    """Convert libcudf unique_ptr[table] to cudf DataFrame."""
    cdef plc_Table plc_table = plc_Table.from_libcudf(move(c_table_ptr))
    columns = [Column.from_pylibcudf(col) for col in plc_table.columns()]

    if index_names:
        n_index = len(index_names)
        index_columns = columns[:n_index]
        data_columns = columns[n_index:]
        index = cudf.MultiIndex._from_data(dict(zip(index_names, index_columns)))
        return cudf.DataFrame._from_data(dict(zip(column_names, data_columns)), index=index)
    else:
        return cudf.DataFrame._from_data(dict(zip(column_names, columns)))

def repartition(object input_table, context, object rows_per_worker=None,
                ignore_index=False) -> cudf.DataFrame:
    cdef table_view c_tv = _dataframe_to_table_view(input_table, ignore_index)
    cdef vector[int] c_rows_per_worker
    cdef CStatus status
    cdef shared_ptr[CCylonContext] c_ctx_ptr = pycylon_unwrap_context(context)

    if rows_per_worker:
        c_rows_per_worker = rows_per_worker

    index_names = None if ignore_index else input_table._index_names

    # Perform repartitioning
    cdef unique_ptr[table] c_table_out
    status = Repartition(
        c_tv,
        c_ctx_ptr,
        c_table_out,
        c_rows_per_worker
    )

    if status.is_ok():
        return _table_to_dataframe(c_table_out, input_table._column_names, index_names)
    else:
        raise ValueError(f"Repartition operation failed : {status.get_msg().decode()}")

def gather(object input_table, context, object gather_root,
           ignore_index=False, ) -> cudf.DataFrame:
    cdef table_view c_tv = _dataframe_to_table_view(input_table, ignore_index)
    cdef int c_gather_root = gather_root
    cdef CStatus c_status
    cdef shared_ptr[CCylonContext] c_ctx_ptr = pycylon_unwrap_context(context)

    index_names = None if ignore_index else input_table._index_names

    # Perform gather
    cdef unique_ptr[table] c_table_out
    c_status = Gather(
        c_tv,
        c_ctx_ptr,
        c_table_out,
        c_gather_root,
    )

    if c_status.is_ok():
        return _table_to_dataframe(c_table_out, input_table._column_names, index_names)
    else:
        raise ValueError(f"Gather operation failed : {c_status.get_msg().decode()}")

def allgather(object input_table, context,
              ignore_index=False, ) -> cudf.DataFrame:
    cdef table_view c_tv = _dataframe_to_table_view(input_table, ignore_index)
    cdef CStatus c_status
    cdef shared_ptr[CCylonContext] c_ctx_ptr = pycylon_unwrap_context(context)

    index_names = None if ignore_index else input_table._index_names

    # Perform allgather
    cdef unique_ptr[table] c_table_out
    c_status = AllGather(
        c_tv,
        c_ctx_ptr,
        c_table_out,
    )

    if c_status.is_ok():
        return _table_to_dataframe(c_table_out, input_table._column_names, index_names)
    else:
        raise ValueError(f"AllGather operation failed : {c_status.get_msg().decode()}")

def broadcast(object input_table, context, object root,
              ignore_index=False, ) -> cudf.DataFrame:
    cdef table_view c_tv = _dataframe_to_table_view(input_table, ignore_index)
    cdef int c_root = root
    cdef CStatus c_status
    cdef shared_ptr[CCylonContext] c_ctx_ptr = pycylon_unwrap_context(context)

    index_names = None if ignore_index else input_table._index_names

    # Perform broadcast
    cdef unique_ptr[table] c_table_out
    c_status = Broadcast(
        c_tv,
        c_root,
        c_ctx_ptr,
        c_table_out,
    )

    if c_status.is_ok():
        return _table_to_dataframe(c_table_out, input_table._column_names, index_names)
    else:
        raise ValueError(f"Broadcast operation failed : {c_status.get_msg().decode()}")
