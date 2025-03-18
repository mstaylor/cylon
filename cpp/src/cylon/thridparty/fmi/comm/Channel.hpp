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

#ifndef CYLON_CHANNEL_HPP
#define CYLON_CHANNEL_HPP

#include "cylon/thridparty/fmi/Data.hpp"
#include <string>
#include <map>
#include <memory>
#include "cylon/thridparty/fmi/utils/Function.hpp"
#include "cylon/thridparty/fmi/utils/Common.hpp"
#include "cylon/thridparty/fmi/utils/Backends.hpp"
#include <cstring>

#include <cstddef>
#include <optional>

using raw_func = std::function<void(char*, char*)>;

//! Struct with functions that is passed to channels, additionally contains information on associativitiy / commutativity (provided by the user)
struct raw_function {
    raw_func f; // Overwrites left arg.
    bool associative;
    bool commutative;
};

//! Data that is passed to and from channels
/*!
 * We intentionally use type erasure such that channels do not need to deal about types.
 * However, the communicator interface ensures that len corresponds to the type in buf and users never directly interact with channel_data.
 */
/*
struct channel_data {
    char* buf;
    std::size_t len;
};
*/

#include <memory>
#include <cstddef>

struct channel_data {
    std::shared_ptr<char[]> buf;  // Shared pointer for buffer management
    std::size_t len = 0;          // Size of the buffer

    // Default constructor
    channel_data() = default;

    // Constructor accepting an existing shared_ptr
    //channel_data(std::shared_ptr<char[]> buffer, std::size_t length)
    //        : buf(buffer), len(length) {}

    channel_data(const char* rawBuffer, std::size_t length)
            : buf(std::shared_ptr<char[]>(new char[length], std::default_delete<char[]>())), len(length) {
        std::memcpy(buf.get(), rawBuffer, length);
    }

    // Constructor allocating a new buffer
    explicit channel_data(std::size_t length)
            : buf(std::shared_ptr<char[]>(new char[length], std::default_delete<char[]>())), len(length) {}

    // Provides access to raw char* buffer
    char* get() {
        return buf.get();
    }
};


namespace FMI::Comm {


    //! Interface that defines channel operations. Only provides a few default implementations, the rest is implemented in the specific ClientServer or PeerToPeer channel types.
    class Channel {
    public:
        //! Send data to peer with id dest, must match a recv call
        virtual void send(const channel_data &buf, FMI::Utils::peer_num dest) = 0;

        //! Send data to peer with id dest, must match a recv call (nonblocking)

        virtual void send(const channel_data &buf, FMI::Utils::peer_num dest,
                              std::function<void(FMI::Utils::NbxStatus, const std::string&)> callback) = 0;


        //! Receive data from peer with id src, must match a send call
        virtual void recv(const channel_data &buf, FMI::Utils::peer_num src) = 0;

        //! Receive data from peer with id src, must match a send call
        virtual void recv(const channel_data &buf, FMI::Utils::peer_num src,
                              std::function<void(FMI::Utils::NbxStatus, const std::string&)> callback) = 0;

        virtual Utils::EventProcessStatus channel_event_progress() = 0;

        //! Broadcast data. Buf only needs to contain useful data for root, the buffer is overwritten for all other peers
        virtual void bcast(const channel_data &buf, FMI::Utils::peer_num root);

        virtual void bcast(const channel_data &buf, FMI::Utils::peer_num root, Utils::Mode mode,
                           std::function<void(FMI::Utils::NbxStatus, const std::string&)> callback);

        //! Barrier synchronization collective.
        virtual void barrier() = 0;

        //! Root gathers data from all peers
        /*!
         * Channel provides a default implementation where peers use send for sending the buffers to the root peer. Child classes may create more optimized implementations.
         * @param sendbuf Data that is sent to the root
         * @param recvbuf Buffer to receive data in, only relevant for root. Needs to have a size of (at least) num_peers * sendbuf.size
         */
        virtual void gather(const channel_data &sendbuf, const channel_data &recvbuf, FMI::Utils::peer_num root);


        virtual void gatherv(const channel_data &sendbuf, const channel_data &recvbuf,
                             FMI::Utils::peer_num root, std::vector<std::size_t> recvcounts,
                             const std::vector<int32_t> displs);

        virtual void gatherv(const channel_data &sendbuf, const channel_data &recvbuf,
                             FMI::Utils::peer_num root, std::vector<std::size_t> recvcounts,
                             const std::vector<int32_t> displs,
                             Utils::Mode mode,
                             std::function<void(FMI::Utils::NbxStatus, const std::string&)> callback);


        /**
         * Gather
         * @param sendbuf
         * @param recvbuf
         * @param root
         */
        virtual void allgather(const channel_data &sendbuf, const channel_data &recvbuf, FMI::Utils::peer_num root);

        virtual void allgather(const channel_data &sendbuf, const channel_data &recvbuf, FMI::Utils::peer_num root,
                               Utils::Mode mode,
                               std::function<void(FMI::Utils::NbxStatus, const std::string&)> callback);

        /**
         * Sends all processes variable-sized data
         * @param sendbuf
         * @param recvbuf
         * @param root
         * @param recvcounts
         * @param displs
         */
        virtual void allgatherv(const channel_data &sendbuf, const channel_data &recvbuf, Utils::peer_num root,
                        const std::vector<int32_t> &recvcounts, const std::vector<int32_t> &displs);

        /**
         * Sends all processes variable-sized data
         * @param sendbuf
         * @param recvbuf
         * @param root
         * @param recvcounts
         * @param displs
         */
        virtual void allgatherv(const channel_data &sendbuf, const channel_data &recvbuf, Utils::peer_num root,
                                const std::vector<int32_t> &recvcounts, const std::vector<int32_t> &displs,
                                Utils::Mode mode,
                                std::function<void(FMI::Utils::NbxStatus, const std::string&)> callback);



        //! Scatter data from root to all peers
        /*!
         * Channel provides a default implementation where root sends the relevant slice to all peers, which use recv to receive it.
         * @param sendbuf Only relevant for root, contains the data that is scattered and needs to have a (divisible) size of num_peers * recvbuf.size
         * @param recvbuf Buffer to receive the data (of size sendbuf.size / num_peers), needs to be set by all peers
         */
        virtual void scatter(const channel_data &sendbuf, const channel_data &recvbuf, FMI::Utils::peer_num root);

        //! Apply function f to sendbuf of all peers.
        /*!
         * When f is not commutative / associative, a strict left-to-right evaluation order is guaranteed.
         * Otherwise, the implementation is free to choose the order, but two executions with the same topology should result in the same order / result.
         * @param sendbuf Data that is sent by every peer
         * @param recvbuf Only relevant for root. Needs to have the same size as sendbuf
         * @param f Associativity / Commutativity of f controls choice of algorithm, depending on the channel / channel type
         */
        virtual void reduce(const channel_data &sendbuf, const channel_data &recvbuf, FMI::Utils::peer_num root, raw_function f) = 0;

        //! Apply function f to sendbuf of all peers, make result available to everyone.
        /*!
         * Provides a default implementation with a reduce, followed by a broadcast. Channel types / implementations can provide more optimized ones.
         * The same considerations for the evaluation order (as with reduce) hold
         * @param sendbuf Data that is sent by every peer
         * @param recvbuf Relevant for all peers in contrast to reduce
         * @param f
         */
        virtual void allreduce(const channel_data &&sendbuf, const channel_data &recvbuf, raw_function f);

        //! Inclusive prefix scan, recvbuf / sendbuf needs to be set for all peers
        virtual void scan(const channel_data &sendbuf, const channel_data &recvbuf, raw_function f) = 0;

        //! Helper utility to set peer id, ID needs to be set before first collective operation
        void set_peer_id(FMI::Utils::peer_num num) { peer_id = num; }

        //! Helper utility to set number of peers, needs to be set before first collective operation
        void set_num_peers(FMI::Utils::peer_num num) { num_peers = num; }

        //! Helper utility to set the communicator name, should be set before first collective operation to avoid conflicts with empty communicator name.
        void set_comm_name(std::string communication_name) {comm_name = communication_name; }

        //! Called before communicator is destructed, can be used by channels to clean up (e.g., delete resources)
        /*!
         * Note that we provide an explicit finalize function on purpose (and do not use a virtual destructor),
         * because derived classes may require that some values of parent classes still exist when cleaning up.
         */
        virtual void finalize() {};

        //! Create a new channel with the given config and model params
        /*!
         * This function is only used for the bundled channels, all other should be instantiated outside of the framework and added over the Communicator interface.
         */
        static std::shared_ptr<Channel> get_channel(std::shared_ptr<FMI::Utils::Backends> &backend);


    protected:
        FMI::Utils::peer_num peer_id;
        FMI::Utils::peer_num num_peers;
        //! Can optionally be used by channels to avoid resource conflicts that may occur because of multiple concurrent communicators.
        /*!
         * For instance, the communicator name can be used as a prefix for key or file names.
         * Some channels might not need this because other mechanisms exist, but every channel has to ensure that multiple concurrent communicators work as expected.
         */
        std::string comm_name;



    };

}



#endif //CYLON_CHANNEL_HPP
