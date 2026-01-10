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
from pycylon.ctx.context cimport CCylonContext
from pycylon.api.lib cimport pycylon_unwrap_context

from pygcylon.net.shuffle cimport Shuffle

import cudf
from cudf._lib.column import Column
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.table cimport Table as plc_Table


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


def shuffle(object input_table, hash_columns, ignore_index, context)  -> cudf.DataFrame:
    cdef CStatus status
    cdef vector[int] c_hash_columns
    cdef shared_ptr[CCylonContext] c_ctx_ptr = pycylon_unwrap_context(context)
    cdef table_view input_tview = _dataframe_to_table_view(input_table, ignore_index)
    cdef unique_ptr[table] c_table_out

    if hash_columns:
        c_hash_columns = hash_columns

        status = Shuffle(input_tview, c_hash_columns, c_ctx_ptr, c_table_out)
        if status.is_ok():
            index_names = None if ignore_index else input_table._index_names
            return _table_to_dataframe(c_table_out, input_table._column_names, index_names)
        else:
            raise ValueError(f"Shuffle operation failed : {status.get_msg().decode()}")
    else:
        raise ValueError('Hash columns are not provided')
