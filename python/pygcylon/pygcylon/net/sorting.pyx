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
from libcpp cimport bool as cppbool

from pycylon.ctx.context cimport CCylonContext
from pycylon.api.lib cimport pycylon_unwrap_context
from pygcylon.net.sorting cimport DistributedSort

import cudf
from cudf._lib.column import Column
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.table cimport Table as plc_Table
cimport pylibcudf.libcudf.types as libcudf_types


cdef table_view _dataframe_to_table_view(df, bint ignore_index):
    """Convert cudf DataFrame to libcudf table_view."""
    cdef list plc_columns = []
    cdef plc_Table plc_table
    for col in df._columns:
        plc_columns.append(col.to_pylibcudf(mode="read"))
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


def distributed_sort(
        object tbl,
        context,
        object sort_columns=None,
        ascending=True,
        nulls_after=True,
        ignore_index=False,
        by_index=False,
):
    cdef table_view c_tv = _dataframe_to_table_view(tbl, ignore_index)
    cdef vector[int] c_sort_column_indices
    cdef vector[libcudf_types.order] c_column_orders
    cdef CStatus status
    cdef shared_ptr[CCylonContext] c_ctx_ptr = pycylon_unwrap_context(context)
    cdef cppbool c_ascending = ascending
    cdef cppbool c_nulls_after = nulls_after

    # Determine sort_column indices from index names
    # ref: cudf/python/cudf/merge.pyx
    if ignore_index:
        num_index_columns = 0
        index_names = None
    else:
        num_index_columns = (
            0 if tbl._index is None
            else tbl._index._num_columns
        )
        index_names = tbl._index_names

    # put indices into the C vector
    if not by_index:
        num_indices = len(sort_columns)
        c_sort_column_indices.reserve(num_indices)
        for cname in sort_columns:
            c_sort_column_indices.push_back(
                num_index_columns + tbl._column_names.index(cname)
            )
    else:
        c_sort_column_indices.reserve(num_index_columns)
        for key in range(0, num_index_columns):
            c_sort_column_indices.push_back(key)

    # construct c_column_orders
    # ascending can be either a bool or a list of bools
    c_column_orders.reserve(c_sort_column_indices.size())
    if isinstance(ascending, list):
        for ascend in ascending:
            order = libcudf_types.order.ASCENDING if ascend else libcudf_types.order.DESCENDING
            c_column_orders.push_back(order)
    elif isinstance(ascending, bool):
        for _ in range(c_sort_column_indices.size()):
            order = libcudf_types.order.ASCENDING if ascending else libcudf_types.order.DESCENDING
            c_column_orders.push_back(order)
    else:
        raise ValueError("ascending must be either a bool or a list of bool")

    # Perform sorting
    cdef unique_ptr[table] c_sorted_table
    #    with nogil:
    status = DistributedSort(
        c_tv,
        c_sort_column_indices,
        c_column_orders,
        c_ctx_ptr,
        c_sorted_table,
        c_nulls_after
    )

    if status.is_ok():
        return _table_to_dataframe(c_sorted_table, tbl._column_names, index_names)
    else:
        raise ValueError(f"Sort operation failed : {status.get_msg().decode()}")
