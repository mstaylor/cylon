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

IF CYTHON_FMI:
    from libcpp cimport bool
    from libcpp.memory cimport shared_ptr
    from libcpp.string cimport string

    from pycylon.net.comm_type cimport CCommType
    from pycylon.net.comm_config cimport CommConfig, CCommConfig

    cdef extern from "../../../../cpp/src/cylon/net/fmi/fmi_communicator.hpp" namespace "cylon::net":
        cdef cppclass CFMIConfig "cylon::net::FMIConfig":
            CCommType Type()

            CFMIConfig(int rank, int world_size, string host, int port, int maxtimeout,
                                        bool resolveip, string comm_name, bool nonblocking)

            @staticmethod
            shared_ptr[CFMIConfig] Make(int rank, int world_size, string host, int port, int maxtimeout,
                                        bool resolveip, string comm_name, bool nonblocking);




    cdef class FMIConfig(CommConfig):
        cdef:
            shared_ptr[CFMIConfig] fmi_config_shd_ptr