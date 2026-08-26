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

'''
Run test:
>> pytest -q python/pycylon/test/test_fmi_config.py
'''

from pycylon.net.fmi_config import FMIConfig
from pycylon.net.comm_type import CommType


def test_fmi_config_direct_redis_channel_type():
    """
    Constructs FMIConfig with channel_type="direct-redis" and asserts it routes
    through the channel-type-aware Make overload. The routing is observable via
    channel_type: only the channel-type-aware overload records the requested type,
    while the legacy Make overload leaves it at its "direct" default, so a reverted
    routing fix fails this assertion.
    """
    config = FMIConfig(0, 1, "127.0.0.1", 9999, 5000, False, "test_comm",
                        False, "127.0.0.1", 6379, "test_ns",
                        channel_type="direct-redis")
    assert config.comm_type == CommType.FMI
    assert config.channel_type == "direct-redis"