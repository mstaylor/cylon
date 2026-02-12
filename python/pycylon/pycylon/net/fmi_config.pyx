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

    cdef class FMIConfig:
        """
        FMIConfig Type mapping from libCylon to PyCylon.

        Supports three channel types:
        - "direct": TCP hole punching via TCPunch (primary for Lambda/serverless)
        - "redis": Redis key-value storage backend (baseline comparison)
        - "s3": S3 object storage backend (baseline comparison)
        """
        def __cinit__(self, rank: int, world_size: int, host: str, port: int, maxtimeout: int,
                      resolveip: bool, comm_name: str, nonblocking: bool, redis_host: str,
                      redis_port: int, redis_namespace: str, enableping: bool = False,
                      channel_type: str = "direct", s3_bucket: str = "", s3_region: str = "us-east-1",
                      key_ttl: int = 3600):
            if world_size < 0:
                raise ValueError("Invalid rank/ world size provided")

            # Use channel_type-aware constructor if channel_type is specified
            if channel_type.lower() in ("redis", "s3"):
                self.fmi_config_shd_ptr = CFMIConfig.Make(
                    rank, world_size, channel_type.encode(),
                    host.encode(), port, maxtimeout,
                    comm_name.encode(), nonblocking,
                    redis_host.encode(), redis_port, redis_namespace.encode(),
                    s3_bucket.encode(), s3_region.encode(),
                    key_ttl)
            else:
                # Use legacy direct backend constructor
                self.fmi_config_shd_ptr = CFMIConfig.Make(
                    rank, world_size, host.encode(), port, maxtimeout,
                    resolveip, comm_name.encode(), nonblocking, enableping,
                    redis_host.encode(), redis_port, redis_namespace.encode())

        @property
        def comm_type(self):
            return self.fmi_config_shd_ptr.get().Type()