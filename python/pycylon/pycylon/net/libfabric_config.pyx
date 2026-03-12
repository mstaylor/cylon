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

IF CYTHON_LIBFABRIC:
    from pycylon.net.comm_config cimport CommConfig
    from pycylon.net.libfabric_config cimport CLibfabricConfig

    cdef class LibfabricConfig:
        """
        LibfabricConfig Type mapping from libCylon to PyCylon.

        Uses libfabric (OFI) for high-performance fabric communication.
        Requires Redis for out-of-band address exchange.

        Parameters
        ----------
        world_size : int
            Total number of processes
        redis_host : str
            Redis server hostname
        redis_port : int
            Redis server port
        session_id : str
            Unique session identifier for Redis key isolation
        key_ttl : int, optional
            TTL in seconds for Redis keys (default 3600)
        provider : str, optional
            Libfabric provider name (e.g. "efa", "verbs", "tcp", "" for auto)
        """
        def __cinit__(self, world_size: int,
                      redis_host: str, redis_port: int,
                      session_id: str,
                      key_ttl: int = 3600,
                      provider: str = ""):
            if world_size < 1:
                raise ValueError("world_size must be >= 1")

            self.libfabric_config_shd_ptr = CLibfabricConfig.Make(
                world_size,
                redis_host.encode(), redis_port,
                session_id.encode(),
                key_ttl,
                provider.encode())

        @property
        def comm_type(self):
            return self.libfabric_config_shd_ptr.get().Type()