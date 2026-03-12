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

# Libfabric Backend Unit Tests

include_directories(
    ${CYLON_SOURCE_DIR}/src
    ${CYLON_SOURCE_DIR}/test
    ${CMAKE_CURRENT_BINARY_DIR}/..
)

function(libfabric_add_test TESTNAME)
    set(exec_name "${TESTNAME}")
    add_executable(${exec_name} ${TESTNAME}.cpp)
    target_link_libraries(${exec_name} cylon)
    target_link_libraries(${exec_name} ${GLOG_LIBRARIES})
endfunction(libfabric_add_test)

# Config test — only needs BUILD_CYLON_LIBFABRIC (no Redis or fabric runtime)
if (CYLON_LIBFABRIC)
    message(STATUS "Building Libfabric config tests")

    libfabric_add_test(libfabric_config_test)
    target_link_libraries(libfabric_config_test ${LIBFABRIC_LIB})

    add_test(NAME libfabric_config_test COMMAND libfabric_config_test)
    set_tests_properties(libfabric_config_test PROPERTIES
        LABELS "libfabric"
    )
endif()

# Redis OOB test — needs both BUILD_CYLON_LIBFABRIC and BUILD_CYLON_REDIS
if (CYLON_LIBFABRIC AND CYLON_USE_REDIS)
    message(STATUS "Building Libfabric Redis OOB tests")

    libfabric_add_test(libfabric_redis_oob_test)
    target_link_libraries(libfabric_redis_oob_test ${LIBFABRIC_LIB})
    target_link_libraries(libfabric_redis_oob_test ${REDIS_LIBRARIES})

    add_test(NAME libfabric_redis_oob_test COMMAND libfabric_redis_oob_test)
    set_tests_properties(libfabric_redis_oob_test PROPERTIES
        LABELS "libfabric;redis"
        ENVIRONMENT "CYLON_LOG_LEVEL=WARNING"
    )
endif()