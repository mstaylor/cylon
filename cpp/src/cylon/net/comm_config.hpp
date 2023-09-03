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

#ifndef CYLON_SRC_CYLON_COMM_COMM_CONFIG_H_
#define CYLON_SRC_CYLON_COMM_COMM_CONFIG_H_
#include <string>
#include <unordered_map>
#include <cylon/net/comm_type.hpp>

namespace cylon {
namespace net {
class CommConfig {
 private:
  std::unordered_map<std::string, std::string> config;

 public:
  void AddConfig(const std::string &key, std::string value) {
	this->config.insert(std::pair<std::string, std::string>(key, value));
  }

  std::string GetConfig(const std::string &key) {

      auto iter = this->config.find(key);
      if (iter == this->config.end()) {
          return {};
      }
	return iter->second;
  }
 public:
  virtual CommType Type() = 0;
  virtual ~CommConfig() = default;
};
}  // namespace net
}  // namespace cylon

#endif //CYLON_SRC_CYLON_COMM_COMM_CONFIG_H_
