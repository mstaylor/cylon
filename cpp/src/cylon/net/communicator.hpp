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

#ifndef CYLON_SRC_CYLON_COMM_COMMUNICATOR_H_
#define CYLON_SRC_CYLON_COMM_COMMUNICATOR_H_

#include "cylon/ctx/memory_pool.hpp"
#include "cylon/net/comm_config.hpp"
#include "cylon/net/channel.hpp"
#include "cylon/net/comm_operations.hpp"

namespace cylon {
class CylonContext;
class Table;
class Column;
class Scalar;

namespace net {

class Communicator {
 public:
  Communicator(MemoryPool *pool, int32_t rank, int32_t world_size)
      : rank(rank), world_size(world_size), pool(pool) {}

  virtual ~Communicator() = default;

  virtual int GetRank() const;
  virtual int GetWorldSize() const;
  virtual MemoryPool* GetMemoryPool() const;
  bool IsFinalized() const;

  virtual CommType GetCommType() const = 0;

  virtual std::unique_ptr<Channel> CreateChannel() const = 0;

  virtual void Finalize() = 0;

  virtual void Barrier() = 0;

  virtual Status AllGather(const std::shared_ptr<Table> &table,
                           std::vector<std::shared_ptr<Table>> *out) const = 0;

  virtual Status Gather(const std::shared_ptr<Table> &table,
                        int gather_root,
                        bool gather_from_root,
                        std::vector<std::shared_ptr<Table>> *out) const = 0;

  /**
   * Broadcasts `table` in `bcast_root` rank to every other rank.
   * @param table Input could be NULL in non-root ranks. Those ranks would have the
   *              broadcast result in this shared_ptr
   * @param bcast_root
   * @param ctx CylonContext is required to instantiate tables in non-root ranks
   * @return
   */
  virtual Status Bcast(std::shared_ptr<Table> *table,
                       int bcast_root,
                       const std::shared_ptr<CylonContext> &ctx) const = 0;

  /* Array communications */

  /**
   * Allreduce values at every index on `values`.
   * @param ctx
   * @param values
   * @param reduce_op
   * @param output
   * @param skip_nulls if `true`,
   * @return
   */
  virtual Status AllReduce(const std::shared_ptr<Column> &values,
                           net::ReduceOp reduce_op,
                           std::shared_ptr<Column> *output) const = 0;

  /**
   * Allgather `values`
   * @param values
   * @param output
   * @return
   */
  virtual Status Allgather(const std::shared_ptr<Column> &values,
                           std::vector<std::shared_ptr<Column>> *output) const = 0;

  /* Scalar communications */

  virtual Status AllReduce(const std::shared_ptr<Scalar> &value,
                           net::ReduceOp reduce_op,
                           std::shared_ptr<Scalar> *output) const = 0;

  virtual Status Allgather(const std::shared_ptr<Scalar> &value,
                           std::shared_ptr<Column> *output) const = 0;

  /* Scatter / Reduce (root-delivering collectives) */

  /**
   * Scatter distributes one table per rank from `scatter_root` to every rank.
   * `tables` is meaningful only at `scatter_root`, where it holds exactly
   * `world_size` entries; rank `r` receives `tables[r]` in `*out`. Per-entry row
   * counts may differ — unequal counts express an uneven (scatterv) scatter, the
   * inverse of how Gather returns a std::vector<Table>.
   *
   * @param ctx CylonContext is required to instantiate the received shard table on
   *            non-root ranks (mirrors Bcast, whose receivers likewise lack a schema).
   *
   * Non-pure with a NotImplemented default so backends that do not support
   * scatter (MPI/gloo/libfabric) are unaffected (Open/Closed).
   */
  virtual Status Scatter(const std::vector<std::shared_ptr<Table>> &tables,
                         int scatter_root,
                         const std::shared_ptr<CylonContext> &ctx,
                         std::shared_ptr<Table> *out) const {
    (void) tables;
    (void) scatter_root;
    (void) ctx;
    (void) out;
    return {Code::NotImplemented, "Scatter not supported by this communicator"};
  }

  /**
   * Reduce `values` element-wise across ranks, delivering the result only at
   * `reduce_root`. Non-root `*output` is left empty/undefined per contract.
   * Non-numeric Arrow types return NotImplemented.
   */
  virtual Status Reduce(const std::shared_ptr<Column> &values,
                        net::ReduceOp reduce_op,
                        int reduce_root,
                        std::shared_ptr<Column> *output) const {
    (void) values;
    (void) reduce_op;
    (void) reduce_root;
    (void) output;
    return {Code::NotImplemented, "Reduce not supported by this communicator"};
  }

  virtual Status Reduce(const std::shared_ptr<Scalar> &value,
                        net::ReduceOp reduce_op,
                        int reduce_root,
                        std::shared_ptr<Scalar> *output) const {
    (void) value;
    (void) reduce_op;
    (void) reduce_root;
    (void) output;
    return {Code::NotImplemented, "Reduce not supported by this communicator"};
  }

 protected:
  int rank = -1;
  int world_size = -1;
  MemoryPool *pool;
  bool finalized = false;
};
}
}

#endif //CYLON_SRC_CYLON_COMM_COMMUNICATOR_H_
