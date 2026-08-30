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

#ifndef CYLON_COMMUNICATOR_HPP
#define CYLON_COMMUNICATOR_HPP


#include <cstring>
#include "comm/Channel.hpp"
#include "utils/Backends.hpp"

namespace FMI {


    //! Helper utility to convert a typed function to a raw function without type information.
    //! Applies f element-wise across ALL count = size_in_bytes / sizeof(T) elements of the buffer.
    //! ClientServer::reduce calls this once per peer to combine two whole buffers, so it must
    //! cover every element (reducing N elements is inherently O(N), exactly as MPI/UCC apply an op
    //! across `count`). The previous body reduced only element [0], silently corrupting every
    //! reduce/allreduce over a multi-element buffer; the single-element case is unchanged (count==1).
    template <typename T>
    raw_func convert_to_raw_function(FMI::Utils::Function<T> f, std::size_t size_in_bytes) {
        std::size_t count = size_in_bytes / sizeof(T);
        auto func = [f, count](char* a, char* b) -> void {
            T* dest = reinterpret_cast<T*>(a);
            T* src = reinterpret_cast<T*>(b);
            for (std::size_t k = 0; k < count; k++) {
                dest[k] = f(dest[k], src[k]);
            }
        };
        return func;
    }




    //! Interface that is exposed to the user for interaction with the FMI system.
    class Communicator {
    public:
        /*!
         * @param peer_id ID of the peer in the range [0 .. num_peers - 1]
         * @param num_peers Number of peers participating in the communicator
         * @param config_path Path to the FMI JSON configuration file
         * @param comm_name Name of the communicator, needs to be unique when multiple communicators are used concurrently
         * @param faas_memory Amount of memory (in MiB) that is allocated to the function, used for performance model calculations.
         */
        Communicator(FMI::Utils::peer_num peer_id, FMI::Utils::peer_num num_peers,
                     const std::shared_ptr<FMI::Utils::Backends> &backend, std::string comm_name,
                     std::string redis_host = "", int redis_port = -1, std::string redis_namespace = "",
                     int ttl_seconds = 3600,
                     int s3_retry_initial_ms = 100, int s3_retry_max_ms = 5000);

        //! Finalizes all active channels
        ~Communicator();

        void finalize() {
            channel->finalize();
        }

        bool checkIfOkToReceive(FMI::Utils::peer_num dest, Utils::Mode mode) {
            return channel->checkReceive(dest, mode);
        }

        bool checkIfOkToSend(FMI::Utils::peer_num dest, Utils::Mode mode) {
            return channel->checkSend(dest, mode);
        }


        //! Send buf to peer dest
        template<typename T>
        void send(Comm::Data<T> &buf, FMI::Utils::peer_num dest) {

            auto data = std::make_shared<channel_data>(buf.data(), buf.size_in_bytes(),
                                                       FMI::Comm::noop_deleter);
            channel->send(data, dest);
        }



        template<typename T>
        void send(Comm::Data<T> &buf, FMI::Utils::peer_num dest,
                  FMI::Utils::fmiContext * context,
                  FMI::Utils::Mode mode,
                  std::function<void(FMI::Utils::NbxStatus, const std::string&,
                                     FMI::Utils::fmiContext *)> callback) {
            auto shared = buf.getShared();

            auto data = std::make_shared<channel_data>(buf.data(), buf.size_in_bytes(),
                                                       FMI::Comm::noop_deleter, shared);
            channel->send(data, dest, context, mode, callback);
        }

        //! Receive data from src and store data into the provided buf
        template<typename T>
        void recv(Comm::Data<T> &buf, FMI::Utils::peer_num src) {
            auto data = std::make_shared<channel_data>(buf.data(), buf.size_in_bytes(),
                                                       FMI::Comm::noop_deleter);
            channel->recv(data, src);
        }

        //! Receive data from src and store data into the provided buf
        /*template<typename T>
        void recv(Comm::Data<T> &buf, FMI::Utils::peer_num src,
                      std::function<void(FMI::Utils::NbxStatus, const std::string&, FMI::Utils::fmiContext *)> callback) {
            channel_data data {buf.data(), buf.size_in_bytes(), FMI::Comm::noop_deleter};
            channel->recv(data, src, callback);
        }*/

        //! Receive data from src and store data into the provided buf
        template<typename T>
        void recv(Comm::Data<T> &buf, FMI::Utils::peer_num src,
                  FMI::Utils::fmiContext * context,
                  FMI::Utils::Mode mode,
                  std::function<void(FMI::Utils::NbxStatus, const std::string&, FMI::Utils::fmiContext *)> callback) {
            auto data = std::make_shared<channel_data>(buf.data(), buf.size_in_bytes(),
                                                       FMI::Comm::noop_deleter);
            channel->recv(data, src, context, mode, std::move(callback));
        }

        //! Broadcast the data that is in the provided buf of the root peer. Result is stored in buf for all peers.
        template<typename T>
        void bcast(Comm::Data<T> &buf, FMI::Utils::peer_num root) {
            auto data = std::make_shared<channel_data>(buf.data(), buf.size_in_bytes(),
                                                       FMI::Comm::noop_deleter);
            channel->bcast(data, root);
        }


        //! Broadcast (non-blocking) the data that is in the provided buf of the root peer. Result is stored in buf for all peers.
        template<typename T>
        void bcast(Comm::Data<T> &buf, FMI::Utils::peer_num root, Utils::Mode mode,
                       std::function<void(FMI::Utils::NbxStatus, const std::string&,
                                          FMI::Utils::fmiContext *)> callback) {
            auto data = std::make_shared<channel_data>(buf.data(), buf.size_in_bytes(),
                                                       FMI::Comm::noop_deleter);
            channel->bcast(data, root, mode, callback);
        }


        Utils::EventProcessStatus communicator_event_progress(Utils::Operation op) {
            return channel->channel_event_progress(op);
        }

        //! Barrier synchronization collective
        void barrier() {
            channel->barrier();
        }

        void barrier(Utils::Mode mode) {
            channel->barrier(mode, [](FMI::Utils::NbxStatus, const std::string &,
                                       FMI::Utils::fmiContext *) {});
        }

        //! Gather the data of the individuals peers (in sendbuf) into the recvbuf of root.
        /*!
         * @param sendbuf Data to send to root, needs to be the same size for all peers.
         * @param recvbuf Receive buffer, only relevant for the root process. Size needs to be num_peers * sendbuf.size
         */
        template<typename T>
        void gather(Comm::Data<T> &sendbuf, Comm::Data<T> &recvbuf, FMI::Utils::peer_num root) {
            auto senddata = std::make_shared<channel_data>(sendbuf.data(),
                                                           sendbuf.size_in_bytes(),
                                                           FMI::Comm::noop_deleter);
            auto recvdata = std::make_shared<channel_data>(recvbuf.data(), recvbuf.size_in_bytes(),
                                                           FMI::Comm::noop_deleter);
            channel->gather(senddata, recvdata, root);
        }



        //! Gather the data of the individuals peers (in sendbuf) into the recvbuf of root.
        /*!
         * @param sendbuf Data to send to root, needs to be the same size for all peers.
         * @param recvbuf Receive buffer, only relevant for the root process. Size needs to be num_peers * sendbuf.size
         */
        template<typename T>
        void gatherv(Comm::Data<T> &sendbuf, Comm::Data<T> &recvbuf, FMI::Utils::peer_num root,
                         const std::vector<int32_t> &recvcounts,
                        const std::vector<int32_t> &displs) {
            auto senddata = std::make_shared<channel_data>(sendbuf.data(), sendbuf.size_in_bytes(),
                                                           FMI::Comm::noop_deleter);
            auto recvdata = std::make_shared<channel_data>(recvbuf.data(), recvbuf.size_in_bytes(),
                                                           FMI::Comm::noop_deleter);
            channel->gatherv(senddata, recvdata, root, recvcounts, displs);
        }

        /*!
        * @param sendbuf Data to send to root, needs to be the same size for all peers.
        * @param recvbuf Receive buffer, only relevant for the root process. Size needs to be num_peers * sendbuf.size
        */
        template<typename T>
        void gatherv(Comm::Data<T> &sendbuf, Comm::Data<T> &recvbuf, FMI::Utils::peer_num root,
                     const std::vector<int32_t> &recvcounts,
                     const std::vector<int32_t> &displs, Utils::Mode mode,
                         std::function<void(FMI::Utils::NbxStatus, const std::string&,
                                            FMI::Utils::fmiContext *)> callback) {
            auto senddata = std::make_shared<channel_data>(sendbuf.data(), sendbuf.size_in_bytes(),
                                                           FMI::Comm::noop_deleter);
            auto recvdata = std::make_shared<channel_data>(recvbuf.data(), recvbuf.size_in_bytes(),
                                                           FMI::Comm::noop_deleter);
            channel->gatherv(senddata, recvdata, root, recvcounts, displs, mode, callback);
        }

        //! Gather the data of the individuals peers (in sendbuf) into the recvbuf of root.
        /*!
         * @param sendbuf Data to send to root, needs to be the same size for all peers.
         * @param recvbuf Receive buffer, only relevant for the root process. Size needs to be num_peers * sendbuf.size
         */
        template<typename T>
        void allgather(Comm::Data<T> &sendbuf, Comm::Data<T> &recvbuf, FMI::Utils::peer_num root) {
            auto senddata = std::make_shared<channel_data>(sendbuf.data(), sendbuf.size_in_bytes(),
                                                           FMI::Comm::noop_deleter);
            auto  recvdata = std::make_shared<channel_data>(recvbuf.data(), recvbuf.size_in_bytes(),
                                                            FMI::Comm::noop_deleter);
            channel->allgather(senddata, recvdata, root);
        }


        /*!
       * @param sendbuf Data to send to root, needs to be the same size for all peers.
       * @param recvbuf Receive buffer, only relevant for the root process. Size needs to be num_peers * sendbuf.size
       */
        template<typename T>
        void allgatherv(Comm::Data<T> &sendbuf, Comm::Data<T> &recvbuf, FMI::Utils::peer_num root,
                     std::vector<int32_t> recvcounts,
                        const std::vector<int32_t> displs,
                     Utils::Mode mode,
                     std::function<void(FMI::Utils::NbxStatus, const std::string&,
                                        FMI::Utils::fmiContext *)> callback) {
            auto senddata = std::make_shared<channel_data>(sendbuf.data(), sendbuf.size_in_bytes(),
                                                           FMI::Comm::noop_deleter);
            auto recvdata = std::make_shared<channel_data>(recvbuf.data(), recvbuf.size_in_bytes(),
                                                           FMI::Comm::noop_deleter);
            channel->allgatherv(senddata, recvdata, root,
                                                       recvcounts, displs, mode, callback);
        }



        //! Scatter the data from root's sendbuf to the recvbuf of all peers.
        /*!
         * @param sendbuf The data to scatter, size needs to be recvbuf.size * num_peers (i.e., divisible by the number of peers). Only relevant for the root peer.
         * @param recvbuf Buffer to receive the data, relevant for all peers.
         */
        template<typename T>
        void scatter(Comm::Data<T> &sendbuf, Comm::Data<T> &recvbuf, FMI::Utils::peer_num root) {
            auto senddata = std::make_shared<channel_data>(sendbuf.data(), sendbuf.size_in_bytes(),
                                                           FMI::Comm::noop_deleter);
            auto recvdata = std::make_shared<channel_data>(recvbuf.data(), recvbuf.size_in_bytes(),
                                                           FMI::Comm::noop_deleter);
            channel->scatter(senddata, recvdata, root);
        }

        //! Scatter variable-sized data from root's sendbuf to the recvbuf of all peers.
        /*!
         * The uneven counterpart of scatter (inverse of gatherv): rank r receives sendcounts[r]
         * bytes from root's sendbuf at offset displs[r]. Counts/displs are byte-granular.
         * @param sendbuf Only relevant for root; size sum(sendcounts).
         * @param recvbuf Buffer of size sendcounts[peer_id], relevant for all peers.
         */
        template<typename T>
        void scatterv(Comm::Data<T> &sendbuf, Comm::Data<T> &recvbuf, FMI::Utils::peer_num root,
                      const std::vector<int32_t> &sendcounts, const std::vector<int32_t> &displs,
                      Utils::Mode mode,
                      std::function<void(FMI::Utils::NbxStatus, const std::string&,
                                         FMI::Utils::fmiContext *)> callback) {
            auto senddata = std::make_shared<channel_data>(sendbuf.data(), sendbuf.size_in_bytes(),
                                                           FMI::Comm::noop_deleter);
            auto recvdata = std::make_shared<channel_data>(recvbuf.data(), recvbuf.size_in_bytes(),
                                                           FMI::Comm::noop_deleter);
            channel->scatterv(senddata, recvdata, root, sendcounts, displs, mode, callback);
        }

        //! Perform a reduction with the reduction function f.
        /*! Depending on the associativity / commutativity of f, a different implementation for the reduction may be used.
         * However, in the same topology, the evaluation order should always be the same, irrespectively of the associativity / commutativitiy.
         * @param sendbuf Data to send, relevant for all peers.
         * @param recvbuf Receive buffer that contains the final result, only relevant for root. Needs to have the same size as the sendbuf.
         */
        template <typename T>
        void reduce(Comm::Data<T> &sendbuf, Comm::Data<T> &recvbuf, FMI::Utils::peer_num root, FMI::Utils::Function<T> f) {
            if (peer_id == root && sendbuf.size_in_bytes() != recvbuf.size_in_bytes()) {
                throw std::runtime_error("Dimensions of send and receive data must match");
            }
            bool left_to_right = !(f.commutative && f.associative);
            auto senddata = std::make_shared<channel_data>(sendbuf.data(), sendbuf.size_in_bytes(),
                                                           FMI::Comm::noop_deleter);
            auto recvdata = std::make_shared<channel_data>(recvbuf.data(), recvbuf.size_in_bytes(),
                                                           FMI::Comm::noop_deleter);
            auto func = convert_to_raw_function(f, sendbuf.size_in_bytes());
            raw_function raw_f {
                    func,
                    f.associative,
                    f.commutative
            };
            channel->reduce(senddata, recvdata, root, raw_f);
        }

        //! Reduce with a type-erased raw function (parallels the allreduce raw overload).
        /*! Needed when send/recv buffers are Comm::Data<void*> (raw byte buffers), where the
         * typed reduce above cannot deduce T consistently between the data and Function<T>.
         * Delegates to the same native channel->reduce.
         * @param sendbuf Data to send, relevant for all peers.
         * @param recvbuf Receive buffer holding the result, only relevant for root; same size as sendbuf.
         */
        template <typename T>
        void reduce(Comm::Data<T> &sendbuf, Comm::Data<T> &recvbuf, FMI::Utils::peer_num root,
                    bool commutative, bool associative, std::function<void(char *, char *)> func) {
            if (peer_id == root && sendbuf.size_in_bytes() != recvbuf.size_in_bytes()) {
                throw std::runtime_error("Dimensions of send and receive data must match");
            }
            auto senddata = std::make_shared<channel_data>(sendbuf.data(), sendbuf.size_in_bytes(),
                                                           FMI::Comm::noop_deleter);
            auto recvdata = std::make_shared<channel_data>(recvbuf.data(), recvbuf.size_in_bytes(),
                                                           FMI::Comm::noop_deleter);
            raw_function raw_f {
                    func,
                    associative,
                    commutative
            };
            channel->reduce(senddata, recvdata, root, raw_f);
        }


        //! Perform a reduction with the reduction function f and make the result available to all peers.
        /*! Depending on the associativity / commutativity of f, a different implementation for the reduction may be used.
         * However, in the same topology, the evaluation order should always be the same, irrespectively of the associativity / commutativitiy.
         * @param sendbuf Data to send, relevant for all peers.
         * @param recvbuf Receive buffer that contains the final result, relevant for all peers. Needs to have the same size as the sendbuf.
         */
        template <typename T>
        void allreduce(Comm::Data<T> &sendbuf, Comm::Data<T> &recvbuf,
                       bool commutative,
                       bool associative,
                       std::function<void(char *,char *)> func,
                       Utils::Mode mode = Utils::BLOCKING) {
            if (sendbuf.size_in_bytes() != recvbuf.size_in_bytes()) {
                throw std::runtime_error("Dimensions of send and receive data must match");
            }
            bool left_to_right = !(commutative && associative);
            auto senddata = std::make_shared<channel_data>(sendbuf.data(), sendbuf.size_in_bytes(),
                                                           FMI::Comm::noop_deleter);
            auto recvdata = std::make_shared<channel_data>(recvbuf.data(), recvbuf.size_in_bytes(),
                                                           FMI::Comm::noop_deleter);
            //auto func = convert_to_raw_function(f, sendbuf.size_in_bytes());
            raw_function raw_f {
                    func,
                    associative,
                    commutative
            };
            if (mode == Utils::BLOCKING) {
                channel->allreduce(std::move(senddata), recvdata, raw_f);
            } else {
                channel->allreduce(std::move(senddata), recvdata, raw_f, mode,
                                   [](FMI::Utils::NbxStatus, const std::string &,
                                      FMI::Utils::fmiContext *) {});
            }
        }


        //! Perform a reduction with the reduction function f and make the result available to all peers.
        /*! Depending on the associativity / commutativity of f, a different implementation for the reduction may be used.
         * However, in the same topology, the evaluation order should always be the same, irrespectively of the associativity / commutativitiy.
         * @param sendbuf Data to send, relevant for all peers.
         * @param recvbuf Receive buffer that contains the final result, relevant for all peers. Needs to have the same size as the sendbuf.
         */
        template <typename T>
        void allreduce(Comm::Data<T> &sendbuf, Comm::Data<T> &recvbuf, FMI::Utils::Function<T> f) {
            if (sendbuf.size_in_bytes() != recvbuf.size_in_bytes()) {
                throw std::runtime_error("Dimensions of send and receive data must match");
            }
            bool left_to_right = !(f.commutative && f.associative);
            auto senddata = std::make_shared<channel_data>(sendbuf.data(), sendbuf.size_in_bytes(),
                                                           FMI::Comm::noop_deleter);
            auto recvdata = std::make_shared<channel_data>(recvbuf.data(), recvbuf.size_in_bytes(),
                                                           FMI::Comm::noop_deleter);
            auto func = convert_to_raw_function(f, sendbuf.size_in_bytes());
            raw_function raw_f {
                    func,
                    f.associative,
                    f.commutative
            };
            channel->allreduce(std::move(senddata), recvdata, raw_f);
        }

        //! Inclusive prefix scan.
        /*! Depending on the associativity / commutativity of f, a different implementation for the reduction may be used.
         * However, in the same topology, the evaluation order should always be the same, irrespectively of the associativity / commutativitiy.
         * @param sendbuf Data to send, relevant for all peers.
         * @param recvbuf Receive buffer that contains the final result, relevant for all peers. Needs to have the same size as the sendbuf.
         */
        template<typename T>
        void scan(Comm::Data<T> &sendbuf, Comm::Data<T> &recvbuf, FMI::Utils::Function<T> f) {
            if (sendbuf.size_in_bytes() != recvbuf.size_in_bytes()) {
                throw std::runtime_error("Dimensions of send and receive data must match");
            }
            auto senddata = std::make_shared<channel_data>(sendbuf.data(), sendbuf.size_in_bytes(),
                                                           FMI::Comm::noop_deleter);
            auto recvdata = std::make_shared<channel_data>(recvbuf.data(), recvbuf.size_in_bytes(),
                                                           FMI::Comm::noop_deleter);
            auto func = convert_to_raw_function(f, sendbuf.size_in_bytes());
            raw_function raw_f {
                    func,
                    f.associative,
                    f.commutative
            };
            channel->scan(senddata, recvdata, raw_f);
        }

        //! Add a new channel to the communicator with the given name by providing a pointer to it.
        void register_channel(std::string name, std::shared_ptr<FMI::Comm::Channel> c, Utils::Operation op);

        Utils::peer_num getPeerId() const;

        Utils::peer_num getNumPeers() const;

    private:

        std::shared_ptr<FMI::Comm::Channel> channel;
        FMI::Utils::peer_num peer_id;
        FMI::Utils::peer_num num_peers;
        std::string comm_name;
        int s3_retry_initial_ms_ = 100;
        int s3_retry_max_ms_ = 5000;



        //! Helper utility to convert a vector function to a raw function that operates directly on memory pointers.
        template <typename A>
        raw_func convert_to_raw_function(FMI::Utils::Function<std::vector<A>> f, std::size_t size_in_bytes) {
            auto func = [f, size_in_bytes](char* a, char* b) -> void {
                std::vector<A> vec_a((A*) a, (A*) (a + size_in_bytes));
                std::vector<A> vec_b((A*) b, (A*) (b + size_in_bytes));
                std::vector<A> res = f(vec_a, vec_b);
                std::memcpy(a, (char*) res.data(), size_in_bytes);
            };
            return func;
        }
    };
}

#endif //CYLON_COMMUNICATOR_HPP
