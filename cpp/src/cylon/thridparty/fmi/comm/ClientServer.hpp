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

#ifndef CYLON_CLIENTSERVER_HPP
#define CYLON_CLIENTSERVER_HPP

#include "Channel.hpp"
#include <map>
#include "../utils/Common.hpp"
#include "../utils/FMIOptions.hpp"



namespace FMI::Comm {
    //! Client-Server channel type
    /*!
     * This class provides optimized collectives for client-server channels and defines the interface that these channels need to implement.
     */
    class ClientServer : public Channel {
    public:
        explicit ClientServer(const std::shared_ptr<FMI::Utils::Backends> &backend);

        std::string process_sends(const channel_data &buf, FMI::Utils::peer_num dest);

        std::string process_received(const channel_data &buf, FMI::Utils::peer_num dest);

        //! Constructs file / key name based on sender and recipient and then uploads the data.
        void send(const channel_data &buf, FMI::Utils::peer_num dest) override;

        void send(const channel_data &buf, FMI::Utils::peer_num dest,
                      std::function<void(FMI::Utils::NbxStatus, const std::string&,
                                         FMI::Utils::fmiContext *)> callback) override;

        void send(const channel_data &buf, FMI::Utils::peer_num dest, FMI::Utils::fmiContext *context,
                  std::function<void(FMI::Utils::NbxStatus, const std::string &,
                                     FMI::Utils::fmiContext *)> callback) override;

        Utils::EventProcessStatus channel_event_progress(Utils::Operation op) override;

        //! Waits until the object with the expected file / key name appears (or a timeout occurs), then downloads it.
        void recv(const channel_data &buf, FMI::Utils::peer_num dest) override;

        void recv(const channel_data &buf, FMI::Utils::peer_num src, FMI::Utils::fmiContext *context,
                  std::function<void(FMI::Utils::NbxStatus, const std::string &,
                                     FMI::Utils::fmiContext *)> callback) override;

        void recv(const channel_data &buf, FMI::Utils::peer_num src,
                      std::function<void(FMI::Utils::NbxStatus, const std::string&,
                                         FMI::Utils::fmiContext *)> callback) override;

        //! Root uploads its data, all other peers download the object
        void bcast(channel_data &buf, FMI::Utils::peer_num root) override;

        //! All peers upload a 1 byte file and wait until num_peers files (associated to this operation based on the file name) exist
        void barrier() override;

        //! All peers upload their data. The root peer downloads these objects and applies the function (as soon as objects become available for associative / commutative functions, left-to-right otherwise)
        void reduce(const channel_data &sendbuf, channel_data &recvbuf, FMI::Utils::peer_num root, raw_function f) override;

        //! All peers upload their data and download the needed files to apply the function. Left-to-right evaluation order is enforced for non-commutative / non-associative functions.
        void scan(const channel_data &sendbuf, channel_data &recvbuf, raw_function f) override;

        //! Function to upload data with a given name / key to the server, needs to be implemented by the channels and should never be invoked directly (use upload instead).
        virtual void upload_object(const channel_data &buf, std::string name) = 0;

        //! Function to download data with a given name / key from the server, needs to be implemented by the channels. Returns true when download was successful, false when file does not exist.
        virtual bool download_object(const channel_data &buf, std::string name) = 0;

        //! Try the download (using download_object) until the object appears or the timeout was reached.
        virtual void download(const channel_data &buf, std::string name);

        //! Try the download (using download_object) until the object appears or the timeout was reached.
        virtual void download_nbx(const channel_data &buf, std::string name,
                                  std::function<void(FMI::Utils::NbxStatus, const std::string&,
                                                     FMI::Utils::fmiContext *)> callback);

        //! Uploads objects and keeps track of them.
        virtual void upload(const channel_data &buf, std::string name);

        //! Uploads objects and keeps track of them.
        virtual void upload_nbx(const channel_data &buf, std::string name,
                                std::function<void(FMI::Utils::NbxStatus, const std::string&,
                                                   FMI::Utils::fmiContext *)> callback);

        //! List all the currently existing objects, needs to be implemented by channels. Needed by some collectives that check for the existence of files, but do not care about their content.
        virtual std::vector<std::string> get_object_names() = 0;

        //! Delete the object with the given name, needs to be implemented by channels.
        virtual void delete_object(std::string name) = 0;

        //! Deletes all objects that were created during the execution.
        void finalize() override;


    protected:
        //! Ensures that there are no file / key name conflicts when a collective operation is used multiple times, values are integrated into the file / key name for these operations.
        std::map<std::string, unsigned int> num_operations = {
                {"bcast", 0},
                {"barrier", 0},
                {"reduce", 0},
                {"scan", 0}
        };
        //! Tracks all created objects such that they can be selectively deleted when finalize is called.
        std::vector<std::string> created_objects;
        //! Backoff time when an object does not exist.
        unsigned int timeout;
        //! Maximum time to wait until an object appears, exception is thrown when this value is exceeded.
        unsigned int max_timeout;
    };
}


#endif //CYLON_CLIENTSERVER_HPP
