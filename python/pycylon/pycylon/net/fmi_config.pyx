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
    from pycylon.net.comm_config cimport CommConfig
    from pycylon.net.fmi_config cimport CFMIConfig

    cdef class FMIConfig(CommConfig):
        """
                GlooConfig Type mapping from libCylon to PyCylon
                """
        def __cinit__(self, rank = 0, world_size = 1, host = "cylon-rendezvous.aws-cylondata.com", port = 10000,
                      maxtimeout = 600000, resolveip = false, comm_name = 'fmi_pair'):
            if rank < 0 or world_size < 0:
                raise ValueError(f"Invalid rank/ world size provided")

            self.fmi_config_shd_ptr = CFMIConfig.Make(rank, world_size, host, port, maxtimeout, resolveip, comm_name)

        @property
        def rank(self):
            return self.fmi_config_shd_ptr.get().getRank()

        @property
        def world_size(self):
            return self.fmi_config_shd_ptr.get().getWorldSize()

        @property
        def comm_type(self):
            return self.fmiconfig_shd_ptr.get().Type()