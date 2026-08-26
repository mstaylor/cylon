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

//! Correctness tests for the FMI binomial `scatter_binomial` / `scatterv_binomial`
//! (`src/net/fmi/peer_to_peer.rs`).
//!
//! These validate the *algorithm* (binomial tree, `transform_peer_id` rooting,
//! `tpref` prefix sums, and the reallocate-on-receive working-buffer handling)
//! independently of the real FMI transport. A `MockChannel` routes `send`/`recv`
//! between P in-process threads through a shared `(src, dst)` mailbox, so the tree
//! runs exactly as it would over TCP but with no redis/rendezvous dependency.
//!
//! The oracle is simple: after a scatter rooted at `root`, rank `r` must receive
//! exactly chunk `r` of the root's concatenated send buffer. We check that across
//! several world sizes (including non-powers-of-two) and several roots (including
//! `root != 0`, which exercises the transformed-order wraparound path).
//!
//! Run with: `cargo test --features fmi --test fmi_scatter_binomial_test`

#![cfg(feature = "fmi")]

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::Duration;

use cylon::error::CylonResult;
use cylon::net::fmi::channel::Channel;
use cylon::net::fmi::common::{ChannelData, FmiContext, Mode, NbxCallback, PeerNum, RawFunction};
use cylon::net::fmi::peer_to_peer::{
    scatter_binomial, scatterv_binomial, IOState, PeerToPeerChannel,
};

/// Shared in-process message router keyed by (src, dst).
type Mailbox = Arc<(Mutex<HashMap<(PeerNum, PeerNum), VecDeque<Vec<u8>>>>, Condvar)>;

/// A minimal `PeerToPeerChannel` that moves bytes between threads through a shared
/// mailbox. Only `send`/`recv`/`peer_id`/`num_peers` need real bodies — the
/// binomial scatter/scatterv call nothing else — so every other trait method is a
/// stub.
struct MockChannel {
    peer_id: PeerNum,
    num_peers: PeerNum,
    mail: Mailbox,
}

impl Channel for MockChannel {
    fn set_peer_id(&mut self, v: PeerNum) {
        self.peer_id = v;
    }
    fn set_num_peers(&mut self, v: PeerNum) {
        self.num_peers = v;
    }
    fn set_comm_name(&mut self, _: &str) {}
    fn set_redis_host(&mut self, _: &str) {}
    fn set_redis_port(&mut self, _: i32) {}

    fn peer_id(&self) -> PeerNum {
        self.peer_id
    }
    fn num_peers(&self) -> PeerNum {
        self.num_peers
    }
    fn comm_name(&self) -> &str {
        "mock"
    }

    fn send(&self, buf: Arc<ChannelData>, dest: PeerNum) -> CylonResult<()> {
        let bytes = buf.as_slice()[..].to_vec();
        let (lock, cv) = &*self.mail;
        let mut m = lock.lock().unwrap();
        m.entry((self.peer_id, dest)).or_default().push_back(bytes);
        cv.notify_all();
        Ok(())
    }

    fn recv(&self, buf: Arc<ChannelData>, src: PeerNum) -> CylonResult<()> {
        let (lock, cv) = &*self.mail;
        let mut m = lock.lock().unwrap();
        loop {
            if let Some(q) = m.get_mut(&(src, self.peer_id)) {
                if let Some(bytes) = q.pop_front() {
                    let mut dst = buf.as_mut_slice();
                    let n = bytes.len().min(dst.len());
                    dst[..n].copy_from_slice(&bytes[..n]);
                    return Ok(());
                }
            }
            // Bounded wait so an algorithm bug surfaces as a test failure, not a hang.
            let (g, timeout) = cv.wait_timeout(m, Duration::from_secs(10)).unwrap();
            m = g;
            if timeout.timed_out() {
                let empty = m.get(&(src, self.peer_id)).map_or(true, |q| q.is_empty());
                if empty {
                    panic!("mock recv timeout: rank {} waiting on {}", self.peer_id, src);
                }
            }
        }
    }

    fn send_async(
        &self,
        _: Arc<ChannelData>,
        _: PeerNum,
        _: Option<Arc<FmiContext>>,
        _: Mode,
        _: Option<NbxCallback>,
    ) -> CylonResult<()> {
        unimplemented!("not used by binomial scatter")
    }
    fn recv_async(
        &self,
        _: Arc<ChannelData>,
        _: PeerNum,
        _: Option<Arc<FmiContext>>,
        _: Mode,
        _: Option<NbxCallback>,
    ) -> CylonResult<()> {
        unimplemented!("not used by binomial scatter")
    }
    fn bcast_async(
        &self,
        _: Arc<ChannelData>,
        _: PeerNum,
        _: Mode,
        _: Option<NbxCallback>,
    ) -> CylonResult<()> {
        unimplemented!("not used by binomial scatter")
    }
    fn barrier(&self) -> CylonResult<()> {
        Ok(())
    }
    fn gatherv_async(
        &self,
        _: Arc<ChannelData>,
        _: Arc<ChannelData>,
        _: PeerNum,
        _: &[i32],
        _: &[i32],
        _: Mode,
        _: Option<NbxCallback>,
    ) -> CylonResult<()> {
        unimplemented!("not used by binomial scatter")
    }
    fn allgather_async(
        &self,
        _: Arc<ChannelData>,
        _: Arc<ChannelData>,
        _: PeerNum,
        _: Mode,
        _: Option<NbxCallback>,
    ) -> CylonResult<()> {
        unimplemented!("not used by binomial scatter")
    }
    fn allgatherv_async(
        &self,
        _: Arc<ChannelData>,
        _: Arc<ChannelData>,
        _: PeerNum,
        _: &[i32],
        _: &[i32],
        _: Mode,
        _: Option<NbxCallback>,
    ) -> CylonResult<()> {
        unimplemented!("not used by binomial scatter")
    }
    fn reduce(
        &self,
        _: Arc<ChannelData>,
        _: Arc<ChannelData>,
        _: PeerNum,
        _: &RawFunction,
    ) -> CylonResult<()> {
        unimplemented!("not used by binomial scatter")
    }
    fn scan(
        &self,
        _: Arc<ChannelData>,
        _: Arc<ChannelData>,
        _: &RawFunction,
    ) -> CylonResult<()> {
        unimplemented!("not used by binomial scatter")
    }
}

impl PeerToPeerChannel for MockChannel {
    fn send_object(&self, _: Arc<ChannelData>, _: PeerNum) -> CylonResult<()> {
        unimplemented!("not used by binomial scatter")
    }
    fn send_object_async(
        &self,
        _: Arc<Mutex<IOState>>,
        _: PeerNum,
        _: Mode,
    ) -> CylonResult<()> {
        unimplemented!("not used by binomial scatter")
    }
    fn recv_object(&self, _: Arc<ChannelData>, _: PeerNum) -> CylonResult<()> {
        unimplemented!("not used by binomial scatter")
    }
    fn recv_object_async(
        &self,
        _: Arc<Mutex<IOState>>,
        _: PeerNum,
        _: Mode,
    ) -> CylonResult<()> {
        unimplemented!("not used by binomial scatter")
    }
}

/// Run an even scatter of `p` equal chunks of `s` bytes rooted at `root`, and
/// return each rank's received chunk indexed by rank. Chunk `k` is `s` copies of
/// the byte `k`, so the oracle for rank `r` is `vec![r; s]`.
fn run_scatter(p: PeerNum, s: usize, root: PeerNum) -> Vec<Vec<u8>> {
    let mail: Mailbox = Arc::new((Mutex::new(HashMap::new()), Condvar::new()));

    let handles: Vec<_> = (0..p)
        .map(|r| {
            let mail = mail.clone();
            thread::spawn(move || {
                let ch = MockChannel {
                    peer_id: r,
                    num_peers: p,
                    mail,
                };
                let sendbuf = if r == root {
                    let mut v = Vec::with_capacity((p as usize) * s);
                    for k in 0..p {
                        v.extend(std::iter::repeat(k as u8).take(s));
                    }
                    Arc::new(ChannelData::new(v))
                } else {
                    // Communicator contract: non-root ranks pass an empty send buffer.
                    Arc::new(ChannelData::with_capacity(0))
                };
                let recvbuf = Arc::new(ChannelData::with_capacity(s));
                scatter_binomial(&ch, sendbuf, recvbuf.clone(), root).unwrap();
                let out = recvbuf.as_slice()[..].to_vec();
                out
            })
        })
        .collect();

    handles.into_iter().map(|h| h.join().unwrap()).collect()
}

/// Run a variable-length scatter with the given `sendcounts` (byte counts, in real
/// rank order) rooted at `root`, and return each rank's received chunk. Chunk `k`
/// is `sendcounts[k]` copies of the byte `k`, so the oracle for rank `r` is
/// `vec![r; sendcounts[r]]`.
fn run_scatterv(p: PeerNum, sendcounts: &[i32], root: PeerNum) -> Vec<Vec<u8>> {
    let mail: Mailbox = Arc::new((Mutex::new(HashMap::new()), Condvar::new()));

    let mut displs = vec![0i32; p as usize];
    let mut acc = 0i32;
    for i in 0..p as usize {
        displs[i] = acc;
        acc += sendcounts[i];
    }

    let handles: Vec<_> = (0..p)
        .map(|r| {
            let mail = mail.clone();
            let sendcounts = sendcounts.to_vec();
            let displs = displs.clone();
            thread::spawn(move || {
                let ch = MockChannel {
                    peer_id: r,
                    num_peers: p,
                    mail,
                };
                let sendbuf = if r == root {
                    let mut v = Vec::new();
                    for k in 0..p as usize {
                        v.extend(std::iter::repeat(k as u8).take(sendcounts[k] as usize));
                    }
                    Arc::new(ChannelData::new(v))
                } else {
                    Arc::new(ChannelData::with_capacity(0))
                };
                let my_count = sendcounts[r as usize] as usize;
                let recvbuf = Arc::new(ChannelData::with_capacity(my_count));
                scatterv_binomial(&ch, sendbuf, recvbuf.clone(), root, &sendcounts, &displs)
                    .unwrap();
                let out = recvbuf.as_slice()[..].to_vec();
                out
            })
        })
        .collect();

    handles.into_iter().map(|h| h.join().unwrap()).collect()
}

/// Distinct roots worth testing for a given world size: 0, 1, and the last rank.
fn roots_for(p: PeerNum) -> Vec<PeerNum> {
    let mut roots = vec![0];
    if p > 1 {
        roots.push(1);
        roots.push(p - 1);
    }
    roots.sort_unstable();
    roots.dedup();
    roots
}

#[test]
fn scatter_binomial_delivers_correct_chunk_to_each_rank() {
    let s = 4usize;
    // Include non-powers-of-two (3, 5, 6, 7) to exercise the min(power, ...) edges.
    for &p in &[1, 2, 3, 4, 5, 6, 7, 8] {
        for root in roots_for(p) {
            let got = run_scatter(p, s, root);
            for r in 0..p {
                let expected = vec![r as u8; s];
                assert_eq!(
                    got[r as usize], expected,
                    "scatter p={p} root={root}: rank {r} got {:?}, expected {:?}",
                    got[r as usize], expected
                );
            }
        }
    }
}

#[test]
fn scatterv_binomial_delivers_correct_variable_chunk_to_each_rank() {
    for &p in &[1, 2, 3, 4, 5, 6, 7, 8] {
        // Variable byte counts: rank k gets (k + 1) * 3 bytes.
        let sendcounts: Vec<i32> = (0..p).map(|k| (k + 1) * 3).collect();
        for root in roots_for(p) {
            let got = run_scatterv(p, &sendcounts, root);
            for r in 0..p {
                let expected = vec![r as u8; sendcounts[r as usize] as usize];
                assert_eq!(
                    got[r as usize], expected,
                    "scatterv p={p} root={root}: rank {r} got {:?}, expected {:?}",
                    got[r as usize], expected
                );
            }
        }
    }
}

#[test]
fn scatterv_binomial_handles_zero_length_shards() {
    // Some ranks receive nothing — the tree must still deliver every non-empty shard.
    let p = 6;
    let sendcounts = vec![5, 0, 3, 0, 7, 2];
    for root in roots_for(p) {
        let got = run_scatterv(p, &sendcounts, root);
        for r in 0..p {
            let expected = vec![r as u8; sendcounts[r as usize] as usize];
            assert_eq!(
                got[r as usize], expected,
                "scatterv-zero p={p} root={root}: rank {r} got {:?}, expected {:?}",
                got[r as usize], expected
            );
        }
    }
}