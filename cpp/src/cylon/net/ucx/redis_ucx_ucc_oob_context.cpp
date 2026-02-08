
#include "redis_ucx_ucc_oob_context.hpp"

namespace cylon {
    namespace net {
#ifdef BUILD_CYLON_REDIS

        static std::string get_session_id() {
            const char* sid = getenv("CYLON_SESSION_ID");
            if (!sid || std::string(sid).empty()) {
                throw std::runtime_error(
                    "CYLON_SESSION_ID environment variable not set. "
                    "The launcher must set this to a value shared by all processes to prevent "
                    "Redis key conflicts. Example: export CYLON_SESSION_ID=$(uuidgen)");
            }
            return std::string(sid);
        }

        UCXRedisOOBContext::UCXRedisOOBContext(int ws, std::string rds, int ttl)
                : redis(std::make_shared<sw::redis::Redis>(rds)), world_size(ws),
                  session_id(get_session_id()), ttl_seconds(ttl) {}

        Status UCXRedisOOBContext::InitOOB() { return Status::OK(); };

        Status UCXRedisOOBContext::getWorldSizeAndRank(int &world_size, int &rank) {
            world_size = this->world_size;
            auto key = session_id + ":num_cur_processes";
            int num_cur_processes = redis->incr(key);
            redis->expire(key, std::chrono::seconds(ttl_seconds));
            rank = this->rank = num_cur_processes - 1;

            return Status::OK();
        }

        Status UCXRedisOOBContext::OOBAllgather(uint8_t *src, uint8_t *dst,
                                                size_t srcSize, size_t dstSize) {
            CYLON_UNUSED(dstSize);
            auto addr_key = session_id + ":ucp_worker_addr_mp";
            redis->hset(addr_key, std::to_string(rank),
                        std::string((char *)src, (char *)src + srcSize));
            redis->expire(addr_key, std::chrono::seconds(ttl_seconds));

            auto helper_key = session_id + ":ucx_helper" + std::to_string(rank);
            std::vector<int> v(world_size, 0);
            redis->lpush(helper_key, v.begin(), v.end());
            redis->expire(helper_key, std::chrono::seconds(ttl_seconds));

            for (int i = 0; i < world_size; i++) {
                if (i == rank) continue;
                auto i_str = std::to_string(i);
                auto helperName = session_id + ":ucx_helper" + i_str;

                auto val = redis->hget(addr_key, i_str);
                while (!val) {
                    redis->blpop(helperName);
                    val = redis->hget(addr_key, i_str);
                }

                memcpy(dst + i * srcSize, val.value().data(), srcSize);
            }

            return Status::OK();
        }

        Status UCXRedisOOBContext::Finalize() {
            // Best-effort cleanup of our session keys
            try {
                redis->del(session_id + ":num_cur_processes");
                redis->del(session_id + ":ucp_worker_addr_mp");
                for (int i = 0; i < world_size; i++) {
                    redis->del(session_id + ":ucx_helper" + std::to_string(i));
                }
            } catch (...) {}
            return Status::OK();
        }

        std::shared_ptr<UCXRedisOOBContext> UCXRedisOOBContext::Make(int world_size, std::string redis_addr,
                                                                     int ttl_seconds) {
            return std::make_shared<UCXRedisOOBContext>(world_size, redis_addr, ttl_seconds);
        };

        void UCCRedisOOBContext::InitOOB(int rank) { this->rank = rank; }

        std::shared_ptr<UCXOOBContext> UCCRedisOOBContext::makeUCXOOBContext() {
            return std::make_shared<UCXRedisOOBContext>(world_size, redis_addr, ttl_seconds);
        }

        void *UCCRedisOOBContext::getCollInfo() { return this; }

        ucc_status_t UCCRedisOOBContext::oob_allgather(void *sbuf, void *rbuf,
                                                       size_t msglen, void *coll_info,
                                                       void **req) {
            int world_size = ((UCCRedisOOBContext *)coll_info)->world_size;
            int rank = ((UCCRedisOOBContext *)coll_info)->rank;
            int num_comm = ((UCCRedisOOBContext *)coll_info)->num_oob_allgather;
            ((UCCRedisOOBContext *)coll_info)->num_oob_allgather++;
            auto &session_id = ((UCCRedisOOBContext *)coll_info)->session_id;
            int ttl_seconds = ((UCCRedisOOBContext *)coll_info)->ttl_seconds;

            auto &redis = ((UCCRedisOOBContext *)coll_info)->redis;
            *req = rbuf;
            std::string s((char *)sbuf, ((char *)sbuf) + msglen);

            auto map_key = session_id + ":ucc_oob_mp" + std::to_string(num_comm);
            redis->hset(map_key, std::to_string(rank), s);
            redis->expire(map_key, std::chrono::seconds(ttl_seconds));

            auto signal_key = session_id + ":ucc_helper" + std::to_string(num_comm) + ":" + std::to_string(rank);
            redis->lpush(signal_key, "0");
            redis->expire(signal_key, std::chrono::seconds(ttl_seconds));

            for (int i = 0; i < world_size; i++) {
                if (i == rank) {
                    memcpy((uint8_t*)rbuf + i * msglen, s.data(), msglen);
                } else {
                    auto helperName =
                            session_id + ":ucc_helper" + std::to_string(num_comm) + ":" + std::to_string(i);

                    sw::redis::OptionalString val;
                    do {
                        redis->brpoplpush(helperName, helperName, 0);
                        val = redis->hget(map_key, std::to_string(i));
                    } while (!val);

                    memcpy((uint8_t*)rbuf + i * msglen, val.value().data(), msglen);
                }
            }

            return UCC_OK;
        }

        UCCRedisOOBContext::UCCRedisOOBContext(int ws,
                                               std::string rds, int ttl)
                : world_size(ws), redis(std::make_shared<sw::redis::Redis>(rds)),
                  redis_addr(rds), session_id(get_session_id()), ttl_seconds(ttl) {}

        UCCRedisOOBContext::UCCRedisOOBContext() {
            redis_addr = "tcp://" + std::string(getenv("CYLON_UCX_OOB_REDIS_ADDR"));
            world_size = std::atoi(getenv("CYLON_UCX_OOB_WORLD_SIZE"));
            redis = std::make_shared<sw::redis::Redis>(redis_addr);
            session_id = get_session_id();
            const char* ttl_env = getenv("CYLON_KEY_TTL");
            int ttl_val = ttl_env ? std::atoi(ttl_env) : 0;
            ttl_seconds = (ttl_val > 0) ? ttl_val : 3600;
        }

        ucc_status_t UCCRedisOOBContext::oob_allgather_test(void *req) {
            CYLON_UNUSED(req);
            return UCC_OK;
        }

        ucc_status_t UCCRedisOOBContext::oob_allgather_free(void *req) {
            CYLON_UNUSED(req);
            return UCC_OK;
        }

        OOBType UCCRedisOOBContext::Type() { return OOBType::OOB_REDIS; }

        std::shared_ptr<sw::redis::Redis> UCCRedisOOBContext::getRedis() {
            return this->redis;
        }

        int UCCRedisOOBContext::getWorldSize() { return world_size; }

        void UCCRedisOOBContext::setRank(int rk) { rank = rk; }

        int UCCRedisOOBContext::getRank() { return rank; }

        std::shared_ptr<UCCRedisOOBContext> UCCRedisOOBContext::Make(int world_size, std::string redis_addr,
                                                                     int ttl_seconds) {
            return std::make_shared<UCCRedisOOBContext>(world_size, redis_addr, ttl_seconds);
        }

        Status UCCRedisOOBContext::Finalize() {
            // Best-effort cleanup of our session keys
            try {
                redis->del(session_id + ":num_cur_processes");
                for (int i = 0; i < num_oob_allgather; i++) {
                    redis->del(session_id + ":ucc_oob_mp" + std::to_string(i));
                    for (int j = 0; j < world_size; j++) {
                        redis->del(session_id + ":ucc_helper" + std::to_string(i) + ":" + std::to_string(j));
                    }
                }
            } catch (...) {}
            return Status::OK();
        }

#endif

    }
}