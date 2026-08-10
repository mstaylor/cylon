
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

IF CYTHON_UCX & CYTHON_UCC:


    from pycylon.net.communicator cimport Communicator
    from pycylon.net.ucc_ucx_communicator cimport CUCXUCCCommunicator
    from pycylon.data.scalar cimport Scalar
    from pycylon.data.scalar cimport CScalar
    from pycylon.data.table cimport CTable
    from pycylon.ctx.context cimport CCylonContext
    from libcpp.memory cimport shared_ptr
    from libcpp.vector cimport vector
    from pycylon.net.reduce_op import ReduceOp
    from pycylon.api.lib cimport (pycylon_wrap_table, pycylon_unwrap_table,
                                  pycylon_unwrap_context)
    import pyarrow as pa
    from pyarrow.lib cimport pyarrow_wrap_scalar


    cdef class UCXUCCCommunicator(Communicator):

        def __cinit__(self):
            pass

        cdef void init(self, const shared_ptr[CUCXUCCCommunicator]& communicator):
            self.ucc_cucx_comm_shd_ptr = communicator


        def allreduce(self, value, reduce_op: ReduceOp):
            cdef shared_ptr[CScalar] cresult
            scalarv = Scalar(pa.scalar(value))

            self.ucc_cucx_comm_shd_ptr.get().AllReduce(scalarv.thisPtr, reduce_op, &cresult)

            return pyarrow_wrap_scalar(cresult.get().data()).as_py()

        def reduce(self, value, reduce_op: ReduceOp, root: int):
            # Root-delivering reduce; result only meaningful at `root`.
            cdef shared_ptr[CScalar] cresult
            scalarv = Scalar(pa.scalar(value))
            self.ucc_cucx_comm_shd_ptr.get().Reduce(scalarv.thisPtr, reduce_op, root, &cresult)
            return pyarrow_wrap_scalar(cresult.get().data()).as_py()

        def scatter(self, tables, root: int, context):
            # `tables` (list[Table]) is meaningful only at `root`; rank r receives tables[r].
            cdef vector[shared_ptr[CTable]] c_in
            cdef shared_ptr[CTable] c_out
            cdef shared_ptr[CCylonContext] c_ctx = pycylon_unwrap_context(context)
            for t in tables:
                c_in.push_back(pycylon_unwrap_table(t))
            self.ucc_cucx_comm_shd_ptr.get().Scatter(c_in, root, c_ctx, &c_out)
            return pycylon_wrap_table(c_out)

        def gather(self, table, root: int):
            cdef vector[shared_ptr[CTable]] c_out
            cdef shared_ptr[CTable] c_in = pycylon_unwrap_table(table)
            self.ucc_cucx_comm_shd_ptr.get().Gather(c_in, root, True, &c_out)
            return [pycylon_wrap_table(c_out[i]) for i in range(c_out.size())]

        def allgather(self, table):
            cdef vector[shared_ptr[CTable]] c_out
            cdef shared_ptr[CTable] c_in = pycylon_unwrap_table(table)
            self.ucc_cucx_comm_shd_ptr.get().AllGather(c_in, &c_out)
            return [pycylon_wrap_table(c_out[i]) for i in range(c_out.size())]

        def broadcast(self, table, root: int, context):
            # C++ Bcast mutates the table in place; needs a CylonContext for non-root ranks.
            cdef shared_ptr[CTable] c_tbl = pycylon_unwrap_table(table)
            cdef shared_ptr[CCylonContext] c_ctx = pycylon_unwrap_context(context)
            self.ucc_cucx_comm_shd_ptr.get().Bcast(&c_tbl, root, c_ctx)
            return pycylon_wrap_table(c_tbl)




