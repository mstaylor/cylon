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

from libc.stdint cimport int32_t
from libcpp cimport bool
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.vector cimport vector
from pyarrow.lib cimport CTable as CArrowTable, CSchema as CArrowSchema

from pycylon.ctx.context cimport CCylonContext
from pycylon.common.status cimport CStatus

cdef extern from "cylon/arrow/async_table_exchange.hpp" namespace "cylon":
    cdef cppclass CReceivedTable "cylon::ReceivedTable":
        int source
        int32_t reference
        shared_ptr[CArrowTable] table

    cdef cppclass CAsyncTableExchange "cylon::AsyncTableExchange":
        CAsyncTableExchange(const shared_ptr[CCylonContext] &ctx,
                            shared_ptr[CArrowSchema] schema)
        CStatus Publish(const shared_ptr[CArrowTable] &table, int32_t reference) nogil
        CStatus Drain(vector[CReceivedTable] *out) nogil
        bool Idle()
        CStatus FinishAndDrain(double timeout_s, vector[CReceivedTable] *out) nogil


cdef class AsyncTableExchange:
    cdef:
        unique_ptr[CAsyncTableExchange] exchange
        object context