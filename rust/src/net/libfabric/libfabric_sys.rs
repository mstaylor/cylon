// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! FFI bindings to Libfabric (Open Fabrics Interfaces)
//!
//! These bindings are generated at compile time by build.rs

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(improper_ctypes)]
#![allow(clippy::all)]

// Include the generated bindings
include!(concat!(env!("OUT_DIR"), "/libfabric_bindings.rs"));

// Additional type definitions that may not be generated correctly

/// Context for tracking operations
#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct fi_context {
    pub internal: [*mut ::std::os::raw::c_void; 4usize],
}

/// Extended context for tracking operations (with user data)
#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct fi_context2 {
    pub internal: [*mut ::std::os::raw::c_void; 8usize],
}

// Re-export enum constants with simpler names for compatibility
// Endpoint types
pub const FI_EP_UNSPEC: fi_ep_type = fi_ep_type_FI_EP_UNSPEC;
pub const FI_EP_MSG: fi_ep_type = fi_ep_type_FI_EP_MSG;
pub const FI_EP_DGRAM: fi_ep_type = fi_ep_type_FI_EP_DGRAM;
pub const FI_EP_RDM: fi_ep_type = fi_ep_type_FI_EP_RDM;

// AV types
pub const FI_AV_UNSPEC: fi_av_type = fi_av_type_FI_AV_UNSPEC;
pub const FI_AV_MAP: fi_av_type = fi_av_type_FI_AV_MAP;
pub const FI_AV_TABLE: fi_av_type = fi_av_type_FI_AV_TABLE;

// Progress modes
pub const FI_PROGRESS_UNSPEC: fi_progress = fi_progress_FI_PROGRESS_UNSPEC;
pub const FI_PROGRESS_AUTO: fi_progress = fi_progress_FI_PROGRESS_AUTO;
pub const FI_PROGRESS_MANUAL: fi_progress = fi_progress_FI_PROGRESS_MANUAL;

// CQ formats
pub const FI_CQ_FORMAT_UNSPEC: fi_cq_format = fi_cq_format_FI_CQ_FORMAT_UNSPEC;
pub const FI_CQ_FORMAT_CONTEXT: fi_cq_format = fi_cq_format_FI_CQ_FORMAT_CONTEXT;
pub const FI_CQ_FORMAT_MSG: fi_cq_format = fi_cq_format_FI_CQ_FORMAT_MSG;
pub const FI_CQ_FORMAT_DATA: fi_cq_format = fi_cq_format_FI_CQ_FORMAT_DATA;
pub const FI_CQ_FORMAT_TAGGED: fi_cq_format = fi_cq_format_FI_CQ_FORMAT_TAGGED;

// Wait objects
pub const FI_WAIT_NONE: fi_wait_obj = fi_wait_obj_FI_WAIT_NONE;
pub const FI_WAIT_UNSPEC: fi_wait_obj = fi_wait_obj_FI_WAIT_UNSPEC;
pub const FI_WAIT_SET: fi_wait_obj = fi_wait_obj_FI_WAIT_SET;
pub const FI_WAIT_FD: fi_wait_obj = fi_wait_obj_FI_WAIT_FD;
pub const FI_WAIT_YIELD: fi_wait_obj = fi_wait_obj_FI_WAIT_YIELD;
pub const FI_WAIT_POLLFD: fi_wait_obj = fi_wait_obj_FI_WAIT_POLLFD;

// Address constant
pub const FI_ADDR_UNSPEC: fi_addr_t = !0u64; // (uint64_t) -1

// Atomic operations
pub const FI_MIN: fi_op = fi_op_FI_MIN;
pub const FI_MAX: fi_op = fi_op_FI_MAX;
pub const FI_SUM: fi_op = fi_op_FI_SUM;
pub const FI_PROD: fi_op = fi_op_FI_PROD;
pub const FI_LOR: fi_op = fi_op_FI_LOR;
pub const FI_LAND: fi_op = fi_op_FI_LAND;
pub const FI_BOR: fi_op = fi_op_FI_BOR;
pub const FI_BAND: fi_op = fi_op_FI_BAND;

// Data types - signed
pub const FI_INT8: fi_datatype = fi_datatype_FI_INT8;
pub const FI_INT16: fi_datatype = fi_datatype_FI_INT16;
pub const FI_INT32: fi_datatype = fi_datatype_FI_INT32;
pub const FI_INT64: fi_datatype = fi_datatype_FI_INT64;

// Data types - unsigned
pub const FI_UINT8: fi_datatype = fi_datatype_FI_UINT8;
pub const FI_UINT16: fi_datatype = fi_datatype_FI_UINT16;
pub const FI_UINT32: fi_datatype = fi_datatype_FI_UINT32;
pub const FI_UINT64: fi_datatype = fi_datatype_FI_UINT64;

// Data types - floating point
pub const FI_FLOAT: fi_datatype = fi_datatype_FI_FLOAT;
pub const FI_DOUBLE: fi_datatype = fi_datatype_FI_DOUBLE;

// Helper function to check for EAGAIN
pub fn is_fi_eagain(ret: isize) -> bool {
    ret == -(FI_EAGAIN as isize)
}

// Helper function to check for success
pub fn is_fi_success(ret: i32) -> bool {
    ret == FI_SUCCESS as i32
}

// ============================================================================
// Inline function implementations
// These are implemented as inline functions in the C headers and must be
// reimplemented in Rust since bindgen cannot generate them.
// ============================================================================

use std::ptr;

/// Create a version number (macro in C)
/// #define FI_VERSION(major, minor) (((major) << 16) | (minor))
#[inline]
pub fn FI_VERSION(major: u32, minor: u32) -> u32 {
    (major << 16) | minor
}

/// Allocate a new fi_info structure
/// static inline struct fi_info *fi_allocinfo(void) { return fi_dupinfo(NULL); }
#[inline]
pub unsafe fn fi_allocinfo() -> *mut fi_info {
    fi_dupinfo(ptr::null())
}

/// Close a fabric object
/// static inline int fi_close(struct fid *fid) { return fid->ops->close(fid); }
#[inline]
pub unsafe fn fi_close(fid: *mut fid) -> ::std::os::raw::c_int {
    if fid.is_null() || (*fid).ops.is_null() {
        return -(FI_EINVAL as i32);
    }
    match (*(*fid).ops).close {
        Some(close_fn) => close_fn(fid),
        None => -(FI_ENOSYS as i32),
    }
}

/// Create a domain
/// static inline int fi_domain(struct fid_fabric *fabric, struct fi_info *info,
///                             struct fid_domain **domain, void *context)
/// { return fabric->ops->domain(fabric, info, domain, context); }
#[inline]
pub unsafe fn fi_domain(
    fabric: *mut fid_fabric,
    info: *mut fi_info,
    domain: *mut *mut fid_domain,
    context: *mut ::std::os::raw::c_void,
) -> ::std::os::raw::c_int {
    if fabric.is_null() || (*fabric).ops.is_null() {
        return -(FI_EINVAL as i32);
    }
    match (*(*fabric).ops).domain {
        Some(domain_fn) => domain_fn(fabric, info, domain, context),
        None => -(FI_ENOSYS as i32),
    }
}

/// Open a completion queue
/// static inline int fi_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
///                              struct fid_cq **cq, void *context)
/// { return domain->ops->cq_open(domain, attr, cq, context); }
#[inline]
pub unsafe fn fi_cq_open(
    domain: *mut fid_domain,
    attr: *mut fi_cq_attr,
    cq: *mut *mut fid_cq,
    context: *mut ::std::os::raw::c_void,
) -> ::std::os::raw::c_int {
    if domain.is_null() || (*domain).ops.is_null() {
        return -(FI_EINVAL as i32);
    }
    match (*(*domain).ops).cq_open {
        Some(cq_open_fn) => cq_open_fn(domain, attr, cq, context),
        None => -(FI_ENOSYS as i32),
    }
}

/// Read from a completion queue
/// static inline ssize_t fi_cq_read(struct fid_cq *cq, void *buf, size_t count)
/// { return cq->ops->read(cq, buf, count); }
#[inline]
pub unsafe fn fi_cq_read(
    cq: *mut fid_cq,
    buf: *mut ::std::os::raw::c_void,
    count: usize,
) -> isize {
    if cq.is_null() || (*cq).ops.is_null() {
        return -(FI_EINVAL as isize);
    }
    match (*(*cq).ops).read {
        Some(read_fn) => read_fn(cq, buf, count),
        None => -(FI_ENOSYS as isize),
    }
}

/// Read with wait from a completion queue
#[inline]
pub unsafe fn fi_cq_sread(
    cq: *mut fid_cq,
    buf: *mut ::std::os::raw::c_void,
    count: usize,
    cond: *const ::std::os::raw::c_void,
    timeout: ::std::os::raw::c_int,
) -> isize {
    if cq.is_null() || (*cq).ops.is_null() {
        return -(FI_EINVAL as isize);
    }
    match (*(*cq).ops).sread {
        Some(sread_fn) => sread_fn(cq, buf, count, cond, timeout),
        None => -(FI_ENOSYS as isize),
    }
}

/// Read error from a completion queue
#[inline]
pub unsafe fn fi_cq_readerr(
    cq: *mut fid_cq,
    buf: *mut fi_cq_err_entry,
    flags: u64,
) -> isize {
    if cq.is_null() || (*cq).ops.is_null() {
        return -(FI_EINVAL as isize);
    }
    match (*(*cq).ops).readerr {
        Some(readerr_fn) => readerr_fn(cq, buf, flags),
        None => -(FI_ENOSYS as isize),
    }
}

/// Open an address vector
#[inline]
pub unsafe fn fi_av_open(
    domain: *mut fid_domain,
    attr: *const fi_av_attr,
    av: *mut *mut fid_av,
    context: *mut ::std::os::raw::c_void,
) -> ::std::os::raw::c_int {
    if domain.is_null() || (*domain).ops.is_null() {
        return -(FI_EINVAL as i32);
    }
    match (*(*domain).ops).av_open {
        Some(av_open_fn) => av_open_fn(domain, attr as *mut _, av, context),
        None => -(FI_ENOSYS as i32),
    }
}

/// Insert addresses into an address vector
#[inline]
pub unsafe fn fi_av_insert(
    av: *mut fid_av,
    addr: *const ::std::os::raw::c_void,
    count: usize,
    fi_addr: *mut fi_addr_t,
    flags: u64,
    context: *mut ::std::os::raw::c_void,
) -> ::std::os::raw::c_int {
    if av.is_null() || (*av).ops.is_null() {
        return -(FI_EINVAL as i32);
    }
    match (*(*av).ops).insert {
        Some(insert_fn) => insert_fn(av, addr, count, fi_addr, flags, context),
        None => -(FI_ENOSYS as i32),
    }
}

/// Remove addresses from an address vector
#[inline]
pub unsafe fn fi_av_remove(
    av: *mut fid_av,
    fi_addr: *const fi_addr_t,
    count: usize,
    flags: u64,
) -> ::std::os::raw::c_int {
    if av.is_null() || (*av).ops.is_null() {
        return -(FI_EINVAL as i32);
    }
    match (*(*av).ops).remove {
        Some(remove_fn) => remove_fn(av, fi_addr as *mut _, count, flags),
        None => -(FI_ENOSYS as i32),
    }
}

/// Create an endpoint
#[inline]
pub unsafe fn fi_endpoint(
    domain: *mut fid_domain,
    info: *mut fi_info,
    ep: *mut *mut fid_ep,
    context: *mut ::std::os::raw::c_void,
) -> ::std::os::raw::c_int {
    if domain.is_null() || (*domain).ops.is_null() {
        return -(FI_EINVAL as i32);
    }
    match (*(*domain).ops).endpoint {
        Some(endpoint_fn) => endpoint_fn(domain, info, ep, context),
        None => -(FI_ENOSYS as i32),
    }
}

/// Bind an object to an endpoint
#[inline]
pub unsafe fn fi_ep_bind(
    ep: *mut fid_ep,
    bfid: *mut fid,
    flags: u64,
) -> ::std::os::raw::c_int {
    if ep.is_null() {
        return -(FI_EINVAL as i32);
    }
    let fid_ptr = &mut (*ep).fid as *mut fid;
    if (*fid_ptr).ops.is_null() {
        return -(FI_EINVAL as i32);
    }
    match (*(*fid_ptr).ops).bind {
        Some(bind_fn) => bind_fn(fid_ptr, bfid, flags),
        None => -(FI_ENOSYS as i32),
    }
}

/// Enable an endpoint
#[inline]
pub unsafe fn fi_enable(ep: *mut fid_ep) -> ::std::os::raw::c_int {
    if ep.is_null() {
        return -(FI_EINVAL as i32);
    }
    let fid_ptr = &mut (*ep).fid as *mut fid;
    if (*fid_ptr).ops.is_null() {
        return -(FI_EINVAL as i32);
    }
    match (*(*fid_ptr).ops).control {
        Some(control_fn) => control_fn(fid_ptr, FI_ENABLE as i32, ptr::null_mut()),
        None => -(FI_ENOSYS as i32),
    }
}

/// Get the name/address of a local endpoint
/// Calls ep->cm->getname
#[inline]
pub unsafe fn fi_getname(
    fid: *mut fid,
    addr: *mut ::std::os::raw::c_void,
    addrlen: *mut usize,
) -> ::std::os::raw::c_int {
    // Cast fid to fid_ep to access cm ops
    let ep = fid as *mut fid_ep;
    if ep.is_null() || (*ep).cm.is_null() {
        return -(FI_EINVAL as i32);
    }
    match (*(*ep).cm).getname {
        Some(getname_fn) => getname_fn(fid as fid_t, addr, addrlen),
        None => -(FI_ENOSYS as i32),
    }
}

/// Send data
#[inline]
pub unsafe fn fi_send(
    ep: *mut fid_ep,
    buf: *const ::std::os::raw::c_void,
    len: usize,
    desc: *mut ::std::os::raw::c_void,
    dest_addr: fi_addr_t,
    context: *mut ::std::os::raw::c_void,
) -> isize {
    if ep.is_null() || (*ep).msg.is_null() {
        return -(FI_EINVAL as isize);
    }
    match (*(*ep).msg).send {
        Some(send_fn) => send_fn(ep, buf, len, desc, dest_addr, context),
        None => -(FI_ENOSYS as isize),
    }
}

/// Receive data
#[inline]
pub unsafe fn fi_recv(
    ep: *mut fid_ep,
    buf: *mut ::std::os::raw::c_void,
    len: usize,
    desc: *mut ::std::os::raw::c_void,
    src_addr: fi_addr_t,
    context: *mut ::std::os::raw::c_void,
) -> isize {
    if ep.is_null() || (*ep).msg.is_null() {
        return -(FI_EINVAL as isize);
    }
    match (*(*ep).msg).recv {
        Some(recv_fn) => recv_fn(ep, buf, len, desc, src_addr, context),
        None => -(FI_ENOSYS as isize),
    }
}

/// Tagged send data
#[inline]
pub unsafe fn fi_tsend(
    ep: *mut fid_ep,
    buf: *const ::std::os::raw::c_void,
    len: usize,
    desc: *mut ::std::os::raw::c_void,
    dest_addr: fi_addr_t,
    tag: u64,
    context: *mut ::std::os::raw::c_void,
) -> isize {
    if ep.is_null() || (*ep).tagged.is_null() {
        return -(FI_EINVAL as isize);
    }
    match (*(*ep).tagged).send {
        Some(send_fn) => send_fn(ep, buf, len, desc, dest_addr, tag, context),
        None => -(FI_ENOSYS as isize),
    }
}

/// Tagged receive data
#[inline]
pub unsafe fn fi_trecv(
    ep: *mut fid_ep,
    buf: *mut ::std::os::raw::c_void,
    len: usize,
    desc: *mut ::std::os::raw::c_void,
    src_addr: fi_addr_t,
    tag: u64,
    ignore: u64,
    context: *mut ::std::os::raw::c_void,
) -> isize {
    if ep.is_null() || (*ep).tagged.is_null() {
        return -(FI_EINVAL as isize);
    }
    match (*(*ep).tagged).recv {
        Some(recv_fn) => recv_fn(ep, buf, len, desc, src_addr, tag, ignore, context),
        None => -(FI_ENOSYS as isize),
    }
}

/// Create an AV set
#[inline]
pub unsafe fn fi_av_set(
    av: *mut fid_av,
    attr: *const fi_av_set_attr,
    set: *mut *mut fid_av_set,
    context: *mut ::std::os::raw::c_void,
) -> ::std::os::raw::c_int {
    if av.is_null() || (*av).ops.is_null() {
        return -(FI_EINVAL as i32);
    }
    match (*(*av).ops).av_set {
        Some(av_set_fn) => av_set_fn(av, attr as *mut _, set, context),
        None => -(FI_ENOSYS as i32),
    }
}

/// Insert an address into an AV set
#[inline]
pub unsafe fn fi_av_set_insert(
    set: *mut fid_av_set,
    fi_addr: fi_addr_t,
) -> ::std::os::raw::c_int {
    if set.is_null() || (*set).ops.is_null() {
        return -(FI_EINVAL as i32);
    }
    match (*(*set).ops).insert {
        Some(insert_fn) => insert_fn(set, fi_addr),
        None => -(FI_ENOSYS as i32),
    }
}

/// Get the address of an AV set
#[inline]
pub unsafe fn fi_av_set_addr(
    set: *mut fid_av_set,
    fi_addr: *mut fi_addr_t,
) -> ::std::os::raw::c_int {
    if set.is_null() || (*set).ops.is_null() {
        return -(FI_EINVAL as i32);
    }
    match (*(*set).ops).addr {
        Some(addr_fn) => addr_fn(set, fi_addr),
        None => -(FI_ENOSYS as i32),
    }
}

/// Join a collective group
/// This wraps fi_join with FI_COLLECTIVE flag and fi_collective_addr structure
#[inline]
pub unsafe fn fi_join_collective(
    ep: *mut fid_ep,
    coll_addr: fi_addr_t,
    set: *const fid_av_set,
    flags: u64,
    mc: *mut *mut fid_mc,
    context: *mut ::std::os::raw::c_void,
) -> ::std::os::raw::c_int {
    fi_join(ep, coll_addr, set, flags | (FI_COLLECTIVE as u64), mc, context)
}

/// Join multicast or collective group
/// Calls ep->cm->join
#[inline]
pub unsafe fn fi_join(
    ep: *mut fid_ep,
    coll_addr: fi_addr_t,
    set: *const fid_av_set,
    flags: u64,
    mc: *mut *mut fid_mc,
    context: *mut ::std::os::raw::c_void,
) -> ::std::os::raw::c_int {
    if ep.is_null() || (*ep).cm.is_null() {
        return -(FI_EINVAL as i32);
    }
    // Create fi_collective_addr on stack
    let mut addr = fi_collective_addr {
        set,
        coll_addr,
    };
    match (*(*ep).cm).join {
        Some(join_fn) => join_fn(ep, &mut addr as *mut _ as *const _, flags, mc, context),
        None => -(FI_ENOSYS as i32),
    }
}

/// Barrier collective
#[inline]
pub unsafe fn fi_barrier(
    ep: *mut fid_ep,
    coll_addr: fi_addr_t,
    context: *mut ::std::os::raw::c_void,
) -> isize {
    if ep.is_null() || (*ep).collective.is_null() {
        return -(FI_EINVAL as isize);
    }
    match (*(*ep).collective).barrier {
        Some(barrier_fn) => barrier_fn(ep, coll_addr, context),
        None => -(FI_ENOSYS as isize),
    }
}

/// Broadcast collective
#[inline]
pub unsafe fn fi_broadcast(
    ep: *mut fid_ep,
    buf: *mut ::std::os::raw::c_void,
    count: usize,
    desc: *mut ::std::os::raw::c_void,
    coll_addr: fi_addr_t,
    root_addr: fi_addr_t,
    datatype: fi_datatype,
    flags: u64,
    context: *mut ::std::os::raw::c_void,
) -> isize {
    if ep.is_null() || (*ep).collective.is_null() {
        return -(FI_EINVAL as isize);
    }
    match (*(*ep).collective).broadcast {
        Some(bcast_fn) => bcast_fn(ep, buf, count, desc, coll_addr, root_addr, datatype, flags, context),
        None => -(FI_ENOSYS as isize),
    }
}

/// Allreduce collective
#[inline]
pub unsafe fn fi_allreduce(
    ep: *mut fid_ep,
    buf: *const ::std::os::raw::c_void,
    count: usize,
    desc: *mut ::std::os::raw::c_void,
    result: *mut ::std::os::raw::c_void,
    result_desc: *mut ::std::os::raw::c_void,
    coll_addr: fi_addr_t,
    datatype: fi_datatype,
    op: fi_op,
    flags: u64,
    context: *mut ::std::os::raw::c_void,
) -> isize {
    if ep.is_null() || (*ep).collective.is_null() {
        return -(FI_EINVAL as isize);
    }
    match (*(*ep).collective).allreduce {
        Some(allreduce_fn) => allreduce_fn(ep, buf, count, desc, result, result_desc, coll_addr, datatype, op, flags, context),
        None => -(FI_ENOSYS as isize),
    }
}

/// Allgather collective
#[inline]
pub unsafe fn fi_allgather(
    ep: *mut fid_ep,
    buf: *const ::std::os::raw::c_void,
    count: usize,
    desc: *mut ::std::os::raw::c_void,
    result: *mut ::std::os::raw::c_void,
    result_desc: *mut ::std::os::raw::c_void,
    coll_addr: fi_addr_t,
    datatype: fi_datatype,
    flags: u64,
    context: *mut ::std::os::raw::c_void,
) -> isize {
    if ep.is_null() || (*ep).collective.is_null() {
        return -(FI_EINVAL as isize);
    }
    match (*(*ep).collective).allgather {
        Some(allgather_fn) => allgather_fn(ep, buf, count, desc, result, result_desc, coll_addr, datatype, flags, context),
        None => -(FI_ENOSYS as isize),
    }
}

/// Alltoall collective
#[inline]
pub unsafe fn fi_alltoall(
    ep: *mut fid_ep,
    buf: *const ::std::os::raw::c_void,
    count: usize,
    desc: *mut ::std::os::raw::c_void,
    result: *mut ::std::os::raw::c_void,
    result_desc: *mut ::std::os::raw::c_void,
    coll_addr: fi_addr_t,
    datatype: fi_datatype,
    flags: u64,
    context: *mut ::std::os::raw::c_void,
) -> isize {
    if ep.is_null() || (*ep).collective.is_null() {
        return -(FI_EINVAL as isize);
    }
    match (*(*ep).collective).alltoall {
        Some(alltoall_fn) => alltoall_fn(ep, buf, count, desc, result, result_desc, coll_addr, datatype, flags, context),
        None => -(FI_ENOSYS as isize),
    }
}

/// Reduce collective
#[inline]
pub unsafe fn fi_reduce(
    ep: *mut fid_ep,
    buf: *const ::std::os::raw::c_void,
    count: usize,
    desc: *mut ::std::os::raw::c_void,
    result: *mut ::std::os::raw::c_void,
    result_desc: *mut ::std::os::raw::c_void,
    coll_addr: fi_addr_t,
    root_addr: fi_addr_t,
    datatype: fi_datatype,
    op: fi_op,
    flags: u64,
    context: *mut ::std::os::raw::c_void,
) -> isize {
    if ep.is_null() || (*ep).collective.is_null() {
        return -(FI_EINVAL as isize);
    }
    match (*(*ep).collective).reduce {
        Some(reduce_fn) => reduce_fn(ep, buf, count, desc, result, result_desc, coll_addr, root_addr, datatype, op, flags, context),
        None => -(FI_ENOSYS as isize),
    }
}

/// Scatter collective
#[inline]
pub unsafe fn fi_scatter(
    ep: *mut fid_ep,
    buf: *const ::std::os::raw::c_void,
    count: usize,
    desc: *mut ::std::os::raw::c_void,
    result: *mut ::std::os::raw::c_void,
    result_desc: *mut ::std::os::raw::c_void,
    coll_addr: fi_addr_t,
    root_addr: fi_addr_t,
    datatype: fi_datatype,
    flags: u64,
    context: *mut ::std::os::raw::c_void,
) -> isize {
    if ep.is_null() || (*ep).collective.is_null() {
        return -(FI_EINVAL as isize);
    }
    match (*(*ep).collective).scatter {
        Some(scatter_fn) => scatter_fn(ep, buf, count, desc, result, result_desc, coll_addr, root_addr, datatype, flags, context),
        None => -(FI_ENOSYS as isize),
    }
}

/// Gather collective
#[inline]
pub unsafe fn fi_gather(
    ep: *mut fid_ep,
    buf: *const ::std::os::raw::c_void,
    count: usize,
    desc: *mut ::std::os::raw::c_void,
    result: *mut ::std::os::raw::c_void,
    result_desc: *mut ::std::os::raw::c_void,
    coll_addr: fi_addr_t,
    root_addr: fi_addr_t,
    datatype: fi_datatype,
    flags: u64,
    context: *mut ::std::os::raw::c_void,
) -> isize {
    if ep.is_null() || (*ep).collective.is_null() {
        return -(FI_EINVAL as isize);
    }
    match (*(*ep).collective).gather {
        Some(gather_fn) => gather_fn(ep, buf, count, desc, result, result_desc, coll_addr, root_addr, datatype, flags, context),
        None => -(FI_ENOSYS as isize),
    }
}

/// Reduce-scatter collective
#[inline]
pub unsafe fn fi_reduce_scatter(
    ep: *mut fid_ep,
    buf: *const ::std::os::raw::c_void,
    count: usize,
    desc: *mut ::std::os::raw::c_void,
    result: *mut ::std::os::raw::c_void,
    result_desc: *mut ::std::os::raw::c_void,
    coll_addr: fi_addr_t,
    datatype: fi_datatype,
    op: fi_op,
    flags: u64,
    context: *mut ::std::os::raw::c_void,
) -> isize {
    if ep.is_null() || (*ep).collective.is_null() {
        return -(FI_EINVAL as isize);
    }
    match (*(*ep).collective).reduce_scatter {
        Some(reduce_scatter_fn) => reduce_scatter_fn(ep, buf, count, desc, result, result_desc, coll_addr, datatype, op, flags, context),
        None => -(FI_ENOSYS as isize),
    }
}
