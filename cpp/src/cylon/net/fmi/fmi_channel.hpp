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

#ifndef CYLON_FMI_CHANNEL_HPP
#define CYLON_FMI_CHANNEL_HPP

#include <unordered_map>
#include <queue>

#include "cylon/net/channel.hpp"

namespace cylon {
    namespace net {

        enum FMISendStatus {
            SEND_INIT = 0,
            SEND_LENGTH_POSTED = 1,
            SEND_POSTED = 2,
            SEND_FINISH = 3,
            SEND_DONE = 4
        };

        enum FMIReceiveStatus {
            RECEIVE_INIT = 0,
            RECEIVE_LENGTH_POSTED = 1,
            RECEIVE_POSTED = 2,
            RECEIVED_FIN = 3
        };


        class FMIChannel : public Channel {

        };

    }
}
#endif //CYLON_FMI_CHANNEL_HPP
