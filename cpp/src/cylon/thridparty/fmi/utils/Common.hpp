/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CYLON_COMMON_HPP
#define CYLON_COMMON_HPP

#include <exception>
namespace FMI::Utils {
    //! Type for peer IDs / numbers
    using peer_num = unsigned int;

    //! Custom exception that is thrown on timeouts
    struct Timeout : public std::exception {
        [[nodiscard]] const char * what () const noexcept {
            return "Timeout was reached";
        }
    };

    //! Set by the client, controls the optimization goal of the Channel Policy
    enum Hint {
        fast, cheap
    };

    //! List of currently supported collectives
    enum Operation {
        send, bcast, barrier, gather, scatter, reduce, allreduce, scan
    };

    //! All the information about an operation, passed to the Channel Policy for its decision on which channel to use.
    struct OperationInfo {
        Operation op;
        std::size_t data_size;
        bool left_to_right = false;
    };

}



#endif //CYLON_COMMON_HPP
