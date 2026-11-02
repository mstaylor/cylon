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
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
from pyarrow.lib cimport CTable as CArrowTable, CSchema as CArrowSchema
from pyarrow.lib cimport pyarrow_unwrap_table, pyarrow_wrap_table, pyarrow_unwrap_schema

from pycylon.common.status cimport CStatus
from pycylon.ctx.context cimport CCylonContext
from pycylon.api.lib cimport pycylon_unwrap_context
from pycylon.net.async_table_exchange cimport CAsyncTableExchange, CReceivedTable

'''
Publish-on-completion table exchange from Cylon C++
'''


cdef list _wrap_received(vector[CReceivedTable] &received):
    return [(received[i].source, received[i].reference, pyarrow_wrap_table(received[i].table))
            for i in range(received.size())]


cdef class AsyncTableExchange:
    """
    Publish-on-completion exchange of pyarrow tables between every pair of ranks.

    publish(table, reference) queues the table for every peer and returns.
    drain() progresses the channel and returns [(source, reference, table)] for the
    peers' tables received since the last call. idle() is True once this rank's sends
    have all completed and no table is half received. finish_and_drain(timeout_s)
    waits until every peer has finished and returns the rest, raising RuntimeError
    naming the peers still unfinished when timeout_s passes. Calls must come from the
    thread that owns the context. With a single rank every call is a no-op.

    Parameters
    ----------
    context: CylonContext
    schema: pyarrow.Schema every published table must match
    """

    def __cinit__(self, context, schema):
        cdef shared_ptr[CCylonContext] c_ctx = pycylon_unwrap_context(context)
        cdef shared_ptr[CArrowSchema] c_schema = pyarrow_unwrap_schema(schema)
        self.context = context
        self.exchange.reset(new CAsyncTableExchange(c_ctx, c_schema))

    def publish(self, table, int32_t reference):
        cdef shared_ptr[CArrowTable] c_table = pyarrow_unwrap_table(table)
        cdef CStatus status
        if c_table.get() == NULL:
            raise TypeError(f"publish expects a pyarrow.Table, got {type(table).__name__}")
        with nogil:
            status = self.exchange.get().Publish(c_table, reference)
        if not status.is_ok():
            raise ValueError(f"publish failed: {status.get_msg().decode()}")

    def drain(self):
        cdef vector[CReceivedTable] received
        cdef CStatus status
        with nogil:
            status = self.exchange.get().Drain(&received)
        if not status.is_ok():
            raise RuntimeError(f"drain failed: {status.get_msg().decode()}")
        return _wrap_received(received)

    def idle(self):
        return self.exchange.get().Idle()

    def finish_and_drain(self, double timeout_s):
        cdef vector[CReceivedTable] received
        cdef CStatus status
        with nogil:
            status = self.exchange.get().FinishAndDrain(timeout_s, &received)
        if not status.is_ok():
            raise RuntimeError(f"finish_and_drain failed: {status.get_msg().decode()}")
        return _wrap_received(received)