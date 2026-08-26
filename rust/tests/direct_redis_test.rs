#![cfg(all(feature = "fmi", feature = "redis"))]

use cylon::net::fmi::RedisDirectEstablisher;
use cylon::net::fmi::common::Mode;
use cylon::net::fmi::channel::Channel;
use cylon::net::fmi::common::ChannelData;
use cylon::net::fmi::direct::Direct;
use cylon::net::fmi::DirectBackend;
use std::sync::Arc;

fn redis_host() -> String {
    std::env::var("CYLON_TEST_REDIS_HOST").unwrap_or_else(|_| "127.0.0.1".to_string())
}

fn redis_port() -> i32 {
    std::env::var("CYLON_TEST_REDIS_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(6379)
}

fn unique_comm_name(tag: &str) -> String {
    format!("rust_direct_redis_{}_{}", tag, std::process::id())
}

#[test]
#[ignore = "requires a running Redis server"]
fn publishes_and_looks_up_a_peer_address() {
    let comm_name = unique_comm_name("pubsub");

    let mut rank0 = RedisDirectEstablisher::default();
    rank0.init(
        redis_host(),
        redis_port(),
        String::new(),
        comm_name.clone(),
        0,
        2,
        18991,
        "127.0.0.1".to_string(),
    ).expect("rank 0 init");

    let mut rank1 = RedisDirectEstablisher::default();
    rank1.init(
        redis_host(),
        redis_port(),
        String::new(),
        comm_name.clone(),
        1,
        2,
        18992,
        "127.0.0.1".to_string(),
    ).expect("rank 1 init");

    let found = rank1.lookup_peer_address_for_test(0, 5000).expect("lookup rank 0");
    assert_eq!(found, "127.0.0.1:18991");

    rank0.finalize();
    rank1.finalize();
}

#[test]
#[ignore = "requires a running Redis server"]
fn different_namespaces_do_not_cross_connect() {
    let comm_name = unique_comm_name("namespaced");

    let mut publisher = RedisDirectEstablisher::default();
    publisher.init(
        redis_host(),
        redis_port(),
        "ns-a".to_string(),
        comm_name.clone(),
        0,
        2,
        18993,
        "127.0.0.1".to_string(),
    ).expect("publisher init");

    let mut same_namespace = RedisDirectEstablisher::default();
    same_namespace.init(
        redis_host(),
        redis_port(),
        "ns-a".to_string(),
        comm_name.clone(),
        1,
        2,
        18994,
        "127.0.0.1".to_string(),
    ).expect("same-namespace init");

    let found = same_namespace
        .lookup_peer_address_for_test(0, 5000)
        .expect("same namespace should find the published address");
    assert_eq!(found, "127.0.0.1:18993");

    let mut other_namespace = RedisDirectEstablisher::default();
    other_namespace.init(
        redis_host(),
        redis_port(),
        "ns-b".to_string(),
        comm_name.clone(),
        1,
        2,
        18995,
        "127.0.0.1".to_string(),
    ).expect("other-namespace init");

    let not_found = other_namespace.lookup_peer_address_for_test(0, 1000);
    assert!(
        not_found.is_err(),
        "a different namespace must not see the other namespace's published address"
    );

    publisher.finalize();
    same_namespace.finalize();
    other_namespace.finalize();
}

#[test]
#[ignore = "requires a running Redis server"]
fn lower_rank_listens_and_higher_rank_connects() {
    let comm_name = unique_comm_name("pair");
    let port0 = 18995;
    let port1 = 18996;

    let mut rank0 = RedisDirectEstablisher::default();
    rank0.init(redis_host(), redis_port(), String::new(), comm_name.clone(), 0, 2, port0,
               "127.0.0.1".to_string()).expect("rank 0 init");

    let mut rank1 = RedisDirectEstablisher::default();
    rank1.init(redis_host(), redis_port(), String::new(), comm_name.clone(), 1, 2, port1,
               "127.0.0.1".to_string()).expect("rank 1 init");

    let dialer = std::thread::spawn(move || {
        rank1.connect(1, 0, 5000, Mode::Blocking).map(|_| ()).map_err(|e| e.to_string())
    });

    let accepted = rank0.connect(0, 1, 5000, Mode::Blocking);
    assert!(accepted.is_ok(), "rank 0 failed to accept: {:?}", accepted.err());
    assert!(dialer.join().unwrap().is_ok());

    rank0.finalize();
}

#[test]
#[ignore = "requires a running Redis server"]
fn blocking_and_nonblocking_pairs_stay_distinct() {
    let comm_name = unique_comm_name("modes");
    let port0 = 18997;
    let port1 = 18998;

    let mut rank0 = RedisDirectEstablisher::default();
    rank0.init(redis_host(), redis_port(), String::new(), comm_name.clone(), 0, 2, port0,
               "127.0.0.1".to_string()).expect("rank 0 init");

    let mut rank1 = RedisDirectEstablisher::default();
    rank1.init(redis_host(), redis_port(), String::new(), comm_name.clone(), 1, 2, port1,
               "127.0.0.1".to_string()).expect("rank 1 init");

    let dialer = std::thread::spawn(move || {
        let b = rank1.connect(1, 0, 5000, Mode::Blocking).is_ok();
        let nb = rank1.connect(1, 0, 5000, Mode::NonBlocking).is_ok();
        (b, nb)
    });

    let got_blocking = rank0.connect(0, 1, 5000, Mode::Blocking).is_ok();
    let got_nonblocking = rank0.connect(0, 1, 5000, Mode::NonBlocking).is_ok();

    let (dialed_blocking, dialed_nonblocking) = dialer.join().unwrap();
    assert!(got_blocking && got_nonblocking, "acceptor missed a mode");
    assert!(dialed_blocking && dialed_nonblocking, "dialer missed a mode");

    rank0.finalize();
}

#[test]
#[ignore = "requires a running Redis server"]
fn direct_channel_round_trips_over_direct_redis() {
    let comm_name = unique_comm_name("channel");
    let port0 = 19210;
    let port1 = 19211;

    let make = |listen_port: i32| {
        DirectBackend::new()
            .set_use_direct_redis(true)
            .with_host("127.0.0.1")
            .with_advertise_host("127.0.0.1")
            .with_port(listen_port)
            .with_max_timeout(5000)
            .set_resolve_dns(false)
            .set_blocking_mode(cylon::net::fmi::common::Mode::Blocking)
    };

    let mut direct0 = Direct::new(&make(port0));
    direct0.set_redis_host(&redis_host());
    direct0.set_redis_port(redis_port());
    direct0.set_comm_name(&comm_name);
    direct0.set_peer_id(0);
    direct0.set_num_peers(2);

    let mut direct1 = Direct::new(&make(port1));
    direct1.set_redis_host(&redis_host());
    direct1.set_redis_port(redis_port());
    direct1.set_comm_name(&comm_name);
    direct1.set_peer_id(1);
    direct1.set_num_peers(2);

    direct0.init().expect("rank 0 init");
    direct1.init().expect("rank 1 init");

    let payload = b"hello-from-1-via-redis-direct\0";
    let recv_data = Arc::new(ChannelData::with_capacity(payload.len()));

    std::thread::scope(|s| {
        s.spawn(|| {
            direct0.recv(recv_data.clone(), 1).expect("rank 0 recv");
        });
        s.spawn(|| {
            let buf = Arc::new(ChannelData::from_slice(payload));
            direct1.send(buf, 0).expect("rank 1 send");
        });
    });

    assert_eq!(&recv_data.as_slice()[..], &payload[..]);
}

#[test]
#[ignore = "requires a running Redis server"]
fn four_rank_mesh_round_trips() {
    const NUM_RANKS: i32 = 4;
    let comm_name = unique_comm_name("mesh");
    let base_port = 19400;

    let make = |listen_port: i32| {
        DirectBackend::new()
            .set_use_direct_redis(true)
            .with_host("127.0.0.1")
            .with_advertise_host("127.0.0.1")
            .with_port(listen_port)
            .with_max_timeout(5000)
            .set_resolve_dns(false)
            .set_blocking_mode(cylon::net::fmi::common::Mode::Blocking)
    };

    let mut directs = Vec::new();
    for r in 0..NUM_RANKS {
        let mut d = Direct::new(&make(base_port + r));
        d.set_redis_host(&redis_host());
        d.set_redis_port(redis_port());
        d.set_comm_name(&comm_name);
        d.set_peer_id(r);
        d.set_num_peers(NUM_RANKS);
        d.init().expect("rank init");
        directs.push(d);
    }

    std::thread::scope(|s| {
        for (r, direct) in directs.iter().enumerate() {
            let r = r as i32;
            s.spawn(move || {
                for peer in 0..NUM_RANKS {
                    if peer == r { continue; }
                    let tag = format!("from{}to{}\0", r, peer);
                    let buf = Arc::new(ChannelData::from_slice(tag.as_bytes()));
                    direct.send(buf, peer).expect("mesh send");
                }
                for peer in 0..NUM_RANKS {
                    if peer == r { continue; }
                    let expected = format!("from{}to{}\0", peer, r);
                    let recv_data = Arc::new(ChannelData::with_capacity(expected.len()));
                    direct.recv(recv_data.clone(), peer).expect("mesh recv");
                    assert_eq!(
                        String::from_utf8_lossy(&recv_data.as_slice()[..]),
                        expected,
                        "rank {} got wrong payload from rank {}",
                        r,
                        peer
                    );
                }
            });
        }
    });
}

#[test]
#[ignore = "requires a running Redis server"]
fn nonblocking_mode_establishes_sockets() {
    let comm_name = unique_comm_name("nbmode");
    let port0 = 19600;
    let port1 = 19601;

    let make = |listen_port: i32| {
        DirectBackend::new()
            .set_use_direct_redis(true)
            .with_host("127.0.0.1")
            .with_advertise_host("127.0.0.1")
            .with_port(listen_port)
            .with_max_timeout(5000)
            .set_resolve_dns(false)
            .set_blocking_mode(cylon::net::fmi::common::Mode::NonBlocking)
    };

    let mut direct0 = Direct::new(&make(port0));
    direct0.set_redis_host(&redis_host());
    direct0.set_redis_port(redis_port());
    direct0.set_comm_name(&comm_name);
    direct0.set_peer_id(0);
    direct0.set_num_peers(2);

    let mut direct1 = Direct::new(&make(port1));
    direct1.set_redis_host(&redis_host());
    direct1.set_redis_port(redis_port());
    direct1.set_comm_name(&comm_name);
    direct1.set_peer_id(1);
    direct1.set_num_peers(2);

    std::thread::scope(|s| {
        s.spawn(|| direct0.init().expect("rank 0 nonblocking init"));
        s.spawn(|| direct1.init().expect("rank 1 nonblocking init"));
    });
}

#[test]
#[ignore = "requires a TCPunch rendezvous server at CYLON_TEST_TCPUNCH_PORT"]
fn tcpunch_path_still_round_trips() {
    let comm_name = unique_comm_name("tcpunch_regression");
    let rendezvous_port: i32 = std::env::var("CYLON_TEST_TCPUNCH_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(19000);

    let make = || {
        DirectBackend::new()
            .with_host("127.0.0.1")
            .with_port(rendezvous_port)
            .with_max_timeout(5000)
            .set_resolve_dns(false)
            .set_blocking_mode(cylon::net::fmi::common::Mode::Blocking)
    };

    let mut direct0 = Direct::new(&make());
    direct0.set_comm_name(&comm_name);
    direct0.set_peer_id(0);
    direct0.set_num_peers(2);

    let mut direct1 = Direct::new(&make());
    direct1.set_comm_name(&comm_name);
    direct1.set_peer_id(1);
    direct1.set_num_peers(2);

    direct0.init().expect("rank 0 init");
    direct1.init().expect("rank 1 init");

    let payload = b"hello-over-tcpunch\0";
    let recv_data = Arc::new(ChannelData::with_capacity(payload.len()));

    std::thread::scope(|s| {
        s.spawn(|| {
            direct0.recv(recv_data.clone(), 1).expect("rank 0 recv");
        });
        s.spawn(|| {
            let buf = Arc::new(ChannelData::from_slice(payload));
            direct1.send(buf, 0).expect("rank 1 send");
        });
    });

    assert_eq!(&recv_data.as_slice()[..], &payload[..]);
}
mod direct_redis_backend_tests {
    use cylon::net::fmi::DirectBackend;

    #[test]
    fn direct_backend_defaults_to_tcpunch() {
        let backend = DirectBackend::new();
        assert!(!backend.use_direct_redis());
    }

    #[test]
    fn direct_backend_records_direct_redis_opt_in() {
        let backend = DirectBackend::new().set_use_direct_redis(true);
        assert!(backend.use_direct_redis());
    }
}

mod mode_key_tests {
    use cylon::net::fmi::common::Mode;
    use cylon::net::fmi::redis_direct_pair::peer_and_mode_key;

    #[test]
    fn blocking_and_nonblocking_map_to_distinct_keys() {
        assert_eq!(peer_and_mode_key(1, Mode::Blocking), 2);
        assert_eq!(peer_and_mode_key(1, Mode::NonBlocking), 3);
        assert_ne!(
            peer_and_mode_key(1, Mode::Blocking),
            peer_and_mode_key(1, Mode::NonBlocking)
        );
    }
}

mod ecs_metadata_tests {
    use cylon::net::fmi::redis_direct_pair::{parse_ipv4_from_metadata, split_metadata_uri};
    use cylon::net::fmi::{FMIConfig, RedisDirectEstablisher};

    #[test]
    fn extracts_the_first_ipv4_address() {
        let body = r#"{"Name":"app","Networks":[{"NetworkMode":"awsvpc","IPv4Addresses":["10.0.3.17"]}]}"#;
        assert_eq!(parse_ipv4_from_metadata(body).unwrap(), "10.0.3.17");
    }

    #[test]
    fn rejects_metadata_without_addresses() {
        let body = r#"{"Name":"app","Networks":[{"NetworkMode":"awsvpc"}]}"#;
        assert!(parse_ipv4_from_metadata(body).is_err());
    }

    #[test]
    fn rejects_malformed_json() {
        assert!(parse_ipv4_from_metadata("not json at all").is_err());
    }

    #[test]
    fn splits_a_metadata_uri_into_host_and_path() {
        let (host, port, path) = split_metadata_uri("http://169.254.170.2/v4/abc123").unwrap();
        assert_eq!(host, "169.254.170.2");
        assert_eq!(port, 80);
        assert_eq!(path, "/v4/abc123");
    }

    #[test]
    fn splits_a_metadata_uri_with_an_explicit_port() {
        let (host, port, path) = split_metadata_uri("http://169.254.170.2:51679/v4/abc123").unwrap();
        assert_eq!(host, "169.254.170.2");
        assert_eq!(port, 51679);
        assert_eq!(path, "/v4/abc123");
    }

    #[test]
    fn rejects_a_metadata_uri_with_no_path() {
        assert!(split_metadata_uri("http://169.254.170.2").is_err());
    }

    #[test]
    fn resolve_own_address_falls_through_to_ecs_when_host_override_is_empty() {
        std::env::remove_var("ECS_CONTAINER_METADATA_URI_V4");

        let mut establisher = RedisDirectEstablisher::default();
        establisher.set_listen_port_for_test(19001);

        let result = establisher.resolve_own_address_for_test();

        assert!(result.is_err(), "expected an ECS fallback error, got {:?}", result);
        let message = result.unwrap_err().to_string();
        assert!(
            message.contains("ECS_CONTAINER_METADATA_URI_V4"),
            "expected the ECS-env-var error, got: {}",
            message
        );
    }

    #[test]
    fn default_builder_path_leaves_advertise_host_unset() {
        let config = FMIConfig::builder().host("localhost").build();
        assert!(
            config.backend().advertise_host().is_none(),
            "the default FMIConfigBuilder path must not set advertise_host, \
             or ECS auto-discovery becomes unreachable again"
        );
    }

    #[test]
    #[ignore]
    fn resolve_own_address_uses_ecs_metadata_end_to_end() {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let bound_port = listener.local_addr().unwrap().port();

        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 1024];
            let _ = stream.read(&mut buf);
            let body = r#"{"Networks":[{"IPv4Addresses":["10.0.3.17"]}]}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(response.as_bytes()).unwrap();
        });

        std::env::set_var(
            "ECS_CONTAINER_METADATA_URI_V4",
            format!("http://127.0.0.1:{}/v4/test", bound_port),
        );

        let mut establisher = RedisDirectEstablisher::default();
        establisher.set_listen_port_for_test(19000);

        let resolved = establisher.resolve_own_address_for_test().unwrap();

        std::env::remove_var("ECS_CONTAINER_METADATA_URI_V4");
        server.join().unwrap();

        assert_eq!(resolved, "10.0.3.17:19000");
    }
}

mod ttl_and_cleanup_tests {
    use cylon::net::fmi::redis_direct_pair::addr_ttl_seconds;
    use cylon::net::fmi::RedisDirectEstablisher;

    #[test]
    fn addr_ttl_seconds_reads_the_env_var() {
        std::env::set_var("CYLON_KEY_TTL", "2");
        assert_eq!(addr_ttl_seconds(), 2);
        std::env::remove_var("CYLON_KEY_TTL");
    }

    #[test]
    fn addr_ttl_seconds_defaults_when_unset() {
        std::env::remove_var("CYLON_KEY_TTL");
        assert_eq!(addr_ttl_seconds(), 3600);
    }

    fn redis_host() -> String {
        std::env::var("CYLON_TEST_REDIS_HOST").unwrap_or_else(|_| "127.0.0.1".to_string())
    }

    fn redis_port() -> i32 {
        std::env::var("CYLON_TEST_REDIS_PORT")
            .ok()
            .and_then(|p| p.parse().ok())
            .unwrap_or(6379)
    }

    #[test]
    #[ignore = "requires a running Redis server"]
    fn finalize_removes_the_published_address() {
        let comm_name = format!("rust_direct_redis_finalize_cleanup_{}", std::process::id());

        let mut establisher = RedisDirectEstablisher::default();
        establisher
            .init(
                redis_host(),
                redis_port(),
                String::new(),
                comm_name.clone(),
                0,
                2,
                18999,
                "127.0.0.1".to_string(),
            )
            .expect("init");

        let mut checker = RedisDirectEstablisher::default();
        checker
            .init(
                redis_host(),
                redis_port(),
                String::new(),
                comm_name.clone(),
                1,
                2,
                19099,
                "127.0.0.1".to_string(),
            )
            .expect("checker init");

        let found = checker
            .lookup_peer_address_for_test(0, 5000)
            .expect("address should be published before finalize");
        assert_eq!(found, "127.0.0.1:18999");

        establisher.finalize();

        let after_finalize = checker.lookup_peer_address_for_test(0, 1000);
        assert!(
            after_finalize.is_err(),
            "finalize() should have removed rank 0's published address"
        );

        checker.finalize();
    }
}

mod channel_type_tests {
    use cylon::net::fmi::{ChannelType, FMIConfig};
    use std::str::FromStr;

    #[test]
    fn parses_the_canonical_spelling() {
        assert_eq!(ChannelType::from_str("direct-redis").unwrap(), ChannelType::DirectRedis);
    }

    #[test]
    fn parses_case_insensitively() {
        assert_eq!(ChannelType::from_str("Direct-Redis").unwrap(), ChannelType::DirectRedis);
        assert_eq!(ChannelType::from_str("DIRECT-REDIS").unwrap(), ChannelType::DirectRedis);
    }

    #[test]
    fn parses_the_other_channel_types() {
        assert_eq!(ChannelType::from_str("direct").unwrap(), ChannelType::Direct);
        assert_eq!(ChannelType::from_str("redis").unwrap(), ChannelType::Redis);
        assert_eq!(ChannelType::from_str("s3").unwrap(), ChannelType::S3);
    }

    #[test]
    fn rejects_an_unknown_channel_type_instead_of_falling_back() {
        assert!(ChannelType::from_str("direct-nonTCPunch").is_err());
        assert!(ChannelType::from_str("nonsense").is_err());
    }

    #[test]
    fn builder_opt_in_reaches_the_backend() {
        let config = FMIConfig::builder()
            .rank(0)
            .world_size(2)
            .use_direct_redis(true)
            .build();
        assert!(config.backend().use_direct_redis());
    }

    #[test]
    fn channel_type_round_trips_into_use_direct_redis() {
        let ct = ChannelType::from_str("direct-redis").unwrap();
        let config = FMIConfig::builder()
            .rank(0)
            .world_size(2)
            .channel_type(ct)
            .unwrap()
            .build();
        assert!(config.backend().use_direct_redis());
    }

    #[test]
    fn channel_type_rejects_unsupported_backends() {
        let redis_ct = ChannelType::from_str("redis").unwrap();
        assert!(FMIConfig::builder().channel_type(redis_ct).is_err());
        let s3_ct = ChannelType::from_str("s3").unwrap();
        assert!(FMIConfig::builder().channel_type(s3_ct).is_err());
    }
}
