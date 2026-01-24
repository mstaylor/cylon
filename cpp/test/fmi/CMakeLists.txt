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

# FMI Backend Unit Tests

include_directories(
    ${CYLON_SOURCE_DIR}/src
    ${CYLON_SOURCE_DIR}/test
    ${CMAKE_CURRENT_BINARY_DIR}/..
)

# macro to add a test
function(fmi_add_test TESTNAME)
    set(exec_name "${TESTNAME}")
    add_executable(${exec_name} ${TESTNAME}.cpp)
    target_link_libraries(${exec_name} cylon)
    target_link_libraries(${exec_name} ${GLOG_LIBRARIES})
endfunction(fmi_add_test)

if (CYLON_USE_REDIS)
    message(STATUS "Building Redis FMI tests")

    fmi_add_test(redis_test)
    target_link_libraries(redis_test ${REDIS_LIBRARIES})

    # Add test (requires running Redis server)
    add_test(NAME redis_fmi_test COMMAND redis_test)
    set_tests_properties(redis_fmi_test PROPERTIES
        LABELS "fmi;redis"
        ENVIRONMENT "CYLON_LOG_LEVEL=WARNING"
    )
endif()

if (CYLON_FMI)
    message(STATUS "Building S3 FMI tests")

    fmi_add_test(s3_test)
    target_link_libraries(s3_test ${AWSSDK_LINK_LIBRARIES})

    # Add test (requires AWS credentials and S3 bucket)
    add_test(NAME s3_fmi_test COMMAND s3_test)
    set_tests_properties(s3_fmi_test PROPERTIES
        LABELS "fmi;s3"
        ENVIRONMENT "CYLON_LOG_LEVEL=WARNING"
    )
endif()
