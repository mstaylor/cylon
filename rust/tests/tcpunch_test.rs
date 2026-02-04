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

//! Unit tests for TCPunch Protocol v2

#[cfg(feature = "fmi")]
mod tcpunch_tests {
    use cylon::net::fmi::tcpunch::{
        build_request, parse_response, PairingStatus, PeerInfo,
        CLIENT_REQUEST_SIZE, MAX_PAIRING_NAME, SERVER_RESPONSE_SIZE, TOKEN_LENGTH,
    };
    use std::net::Ipv4Addr;

    // =========================================================================
    // Protocol Constants Tests
    // =========================================================================

    #[test]
    fn test_protocol_constants() {
        // Verify Protocol v2 sizes
        assert_eq!(MAX_PAIRING_NAME, 100);
        assert_eq!(TOKEN_LENGTH, 37);
        assert_eq!(CLIENT_REQUEST_SIZE, 141); // 100 + 37 + 4
        assert_eq!(SERVER_RESPONSE_SIZE, 51); // 1 + 4 + 2 + 4 + 2 + 37 + 1
    }

    // =========================================================================
    // PairingStatus Tests
    // =========================================================================

    #[test]
    fn test_pairing_status_from_u8() {
        assert_eq!(PairingStatus::from(0), PairingStatus::Waiting);
        assert_eq!(PairingStatus::from(1), PairingStatus::Paired);
        assert_eq!(PairingStatus::from(2), PairingStatus::Timeout);
        assert_eq!(PairingStatus::from(3), PairingStatus::Error);
    }

    #[test]
    fn test_pairing_status_unknown_values() {
        // Unknown values should map to Error
        assert_eq!(PairingStatus::from(4), PairingStatus::Error);
        assert_eq!(PairingStatus::from(255), PairingStatus::Error);
    }

    #[test]
    fn test_pairing_status_repr() {
        // Verify repr(u8) values
        assert_eq!(PairingStatus::Waiting as u8, 0);
        assert_eq!(PairingStatus::Paired as u8, 1);
        assert_eq!(PairingStatus::Timeout as u8, 2);
        assert_eq!(PairingStatus::Error as u8, 3);
    }

    // =========================================================================
    // PeerInfo Tests
    // =========================================================================

    #[test]
    fn test_peer_info_to_socket_addr() {
        let peer = PeerInfo {
            ip: Ipv4Addr::new(192, 168, 1, 100),
            port: 8080,
        };
        let addr = peer.to_socket_addr();
        assert_eq!(addr.ip().to_string(), "192.168.1.100");
        assert_eq!(addr.port(), 8080);
    }

    #[test]
    fn test_peer_info_is_empty() {
        let empty = PeerInfo {
            ip: Ipv4Addr::new(0, 0, 0, 0),
            port: 0,
        };
        assert!(empty.is_empty());

        let not_empty = PeerInfo {
            ip: Ipv4Addr::new(192, 168, 1, 1),
            port: 8080,
        };
        assert!(!not_empty.is_empty());

        // Only IP is unspecified
        let only_port = PeerInfo {
            ip: Ipv4Addr::new(0, 0, 0, 0),
            port: 8080,
        };
        assert!(!only_port.is_empty());

        // Only port is zero
        let only_ip = PeerInfo {
            ip: Ipv4Addr::new(192, 168, 1, 1),
            port: 0,
        };
        assert!(!only_ip.is_empty());
    }

    // =========================================================================
    // build_request Tests
    // =========================================================================

    #[test]
    fn test_build_request_size() {
        let request = build_request("test-session", None);
        assert_eq!(request.len(), CLIENT_REQUEST_SIZE);
    }

    #[test]
    fn test_build_request_pairing_name() {
        let request = build_request("my-session-123", None);

        // Check pairing name is at offset 0
        let name = std::str::from_utf8(&request[..14]).unwrap();
        assert_eq!(name, "my-session-123");

        // Rest should be null-terminated
        assert_eq!(request[14], 0);
    }

    #[test]
    fn test_build_request_with_token() {
        let token = "550e8400-e29b-41d4-a716-446655440000";
        let request = build_request("session", Some(token));

        // Check token is at offset 100
        let token_bytes = &request[MAX_PAIRING_NAME..MAX_PAIRING_NAME + 36];
        let extracted = std::str::from_utf8(token_bytes).unwrap();
        assert_eq!(extracted, token);
    }

    #[test]
    fn test_build_request_no_token() {
        let request = build_request("session", None);

        // Token area should be all zeros
        for i in MAX_PAIRING_NAME..MAX_PAIRING_NAME + TOKEN_LENGTH {
            assert_eq!(request[i], 0);
        }
    }

    #[test]
    fn test_build_request_flags_reserved() {
        let request = build_request("session", None);

        // Flags at offset 137 (4 bytes) should be zero
        for i in 137..141 {
            assert_eq!(request[i], 0);
        }
    }

    #[test]
    fn test_build_request_long_name_truncated() {
        let long_name = "a".repeat(150);
        let request = build_request(&long_name, None);

        // Should be truncated to MAX_PAIRING_NAME - 1 = 99 chars
        let name_bytes: Vec<u8> = request[..99].to_vec();
        assert!(name_bytes.iter().all(|&b| b == b'a'));
        assert_eq!(request[99], 0); // Null terminator
    }

    #[test]
    fn test_build_request_long_token_truncated() {
        let long_token = "x".repeat(50);
        let request = build_request("session", Some(&long_token));

        // Should be truncated to TOKEN_LENGTH - 1 = 36 chars
        let token_bytes: Vec<u8> = request[MAX_PAIRING_NAME..MAX_PAIRING_NAME + 36].to_vec();
        assert!(token_bytes.iter().all(|&b| b == b'x'));
    }

    // =========================================================================
    // parse_response Tests
    // =========================================================================

    fn create_response_buffer(
        status: u8,
        your_ip: [u8; 4],
        your_port: u16,
        peer_ip: [u8; 4],
        peer_port: u16,
        token: &str,
    ) -> [u8; SERVER_RESPONSE_SIZE] {
        let mut buf = [0u8; SERVER_RESPONSE_SIZE];

        // Status (offset 0)
        buf[0] = status;

        // Your IP (offset 1-4, network byte order)
        buf[1..5].copy_from_slice(&your_ip);

        // Your port (offset 5-6, network byte order)
        buf[5..7].copy_from_slice(&your_port.to_be_bytes());

        // Peer IP (offset 7-10, network byte order)
        buf[7..11].copy_from_slice(&peer_ip);

        // Peer port (offset 11-12, network byte order)
        buf[11..13].copy_from_slice(&peer_port.to_be_bytes());

        // Token (offset 13-49)
        let token_bytes = token.as_bytes();
        let len = token_bytes.len().min(37);
        buf[13..13 + len].copy_from_slice(&token_bytes[..len]);

        buf
    }

    #[test]
    fn test_parse_response_waiting() {
        let buf = create_response_buffer(
            0, // WAITING
            [192, 168, 1, 100],
            8080,
            [0, 0, 0, 0],
            0,
            "abc-123-token",
        );

        let resp = parse_response(&buf);

        assert_eq!(resp.status, PairingStatus::Waiting);
        assert_eq!(resp.your_info.ip, Ipv4Addr::new(192, 168, 1, 100));
        assert_eq!(resp.your_info.port, 8080);
        assert!(resp.peer_info.is_none());
        assert_eq!(resp.token, "abc-123-token");
    }

    #[test]
    fn test_parse_response_paired() {
        let buf = create_response_buffer(
            1, // PAIRED
            [192, 168, 1, 100],
            8080,
            [10, 0, 0, 5],
            9090,
            "token-xyz",
        );

        let resp = parse_response(&buf);

        assert_eq!(resp.status, PairingStatus::Paired);
        assert_eq!(resp.your_info.ip, Ipv4Addr::new(192, 168, 1, 100));
        assert_eq!(resp.your_info.port, 8080);

        let peer = resp.peer_info.expect("should have peer info");
        assert_eq!(peer.ip, Ipv4Addr::new(10, 0, 0, 5));
        assert_eq!(peer.port, 9090);
    }

    #[test]
    fn test_parse_response_timeout() {
        let buf = create_response_buffer(
            2, // TIMEOUT
            [192, 168, 1, 100],
            8080,
            [0, 0, 0, 0],
            0,
            "reconnect-token",
        );

        let resp = parse_response(&buf);

        assert_eq!(resp.status, PairingStatus::Timeout);
        assert_eq!(resp.token, "reconnect-token");
    }

    #[test]
    fn test_parse_response_error() {
        let buf = create_response_buffer(
            3, // ERROR
            [0, 0, 0, 0],
            0,
            [0, 0, 0, 0],
            0,
            "",
        );

        let resp = parse_response(&buf);

        assert_eq!(resp.status, PairingStatus::Error);
        assert!(resp.token.is_empty());
    }

    #[test]
    fn test_parse_response_unknown_status() {
        let buf = create_response_buffer(
            99, // Unknown status
            [192, 168, 1, 1],
            1234,
            [0, 0, 0, 0],
            0,
            "",
        );

        let resp = parse_response(&buf);

        // Unknown status maps to Error
        assert_eq!(resp.status, PairingStatus::Error);
    }

    #[test]
    fn test_parse_response_peer_info_present() {
        let buf = create_response_buffer(
            1,
            [192, 168, 1, 100],
            8080,
            [172, 16, 0, 1], // Non-zero peer IP
            5000,            // Non-zero peer port
            "token",
        );

        let resp = parse_response(&buf);

        assert!(resp.peer_info.is_some());
        let peer = resp.peer_info.unwrap();
        assert_eq!(peer.ip, Ipv4Addr::new(172, 16, 0, 1));
        assert_eq!(peer.port, 5000);
    }

    #[test]
    fn test_parse_response_peer_info_absent() {
        let buf = create_response_buffer(
            0,
            [192, 168, 1, 100],
            8080,
            [0, 0, 0, 0], // All zeros
            0,
            "token",
        );

        let resp = parse_response(&buf);

        assert!(resp.peer_info.is_none());
    }

    #[test]
    fn test_parse_response_full_token() {
        let full_token = "550e8400-e29b-41d4-a716-446655440000"; // 36 chars
        let buf = create_response_buffer(
            0,
            [192, 168, 1, 100],
            8080,
            [0, 0, 0, 0],
            0,
            full_token,
        );

        let resp = parse_response(&buf);

        assert_eq!(resp.token, full_token);
    }

    #[test]
    fn test_parse_response_empty_token() {
        let buf = create_response_buffer(
            0,
            [192, 168, 1, 100],
            8080,
            [0, 0, 0, 0],
            0,
            "",
        );

        let resp = parse_response(&buf);

        assert!(resp.token.is_empty());
    }

    // =========================================================================
    // Round-trip Tests
    // =========================================================================

    #[test]
    fn test_request_response_roundtrip() {
        // Build a request
        let pairing_name = "test-session-abc";
        let token = "my-reconnect-token";
        let request = build_request(pairing_name, Some(token));

        // Verify request structure
        assert_eq!(request.len(), CLIENT_REQUEST_SIZE);

        // Extract and verify pairing name
        let name_end = request[..MAX_PAIRING_NAME]
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(MAX_PAIRING_NAME);
        let extracted_name = std::str::from_utf8(&request[..name_end]).unwrap();
        assert_eq!(extracted_name, pairing_name);

        // Extract and verify token
        let token_end = request[MAX_PAIRING_NAME..MAX_PAIRING_NAME + TOKEN_LENGTH]
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(TOKEN_LENGTH);
        let extracted_token = std::str::from_utf8(
            &request[MAX_PAIRING_NAME..MAX_PAIRING_NAME + token_end]
        ).unwrap();
        assert_eq!(extracted_token, token);
    }

    // =========================================================================
    // Integration Tests (require TCPunch server)
    // =========================================================================

    #[test]
    #[ignore = "requires TCPunch server running"]
    fn test_pair_connection() {
        // This test requires a TCPunch server to be running
        // Run with: cargo test --features fmi -- --ignored

        use cylon::net::fmi::tcpunch::pair;
        use std::time::Duration;

        // Two threads simulating two peers
        let handle1 = std::thread::spawn(|| {
            pair("integration-test-session", "127.0.0.1", 10000, 30000)
        });

        let handle2 = std::thread::spawn(|| {
            std::thread::sleep(Duration::from_millis(100));
            pair("integration-test-session", "127.0.0.1", 10000, 30000)
        });

        let result1 = handle1.join().expect("thread 1 panicked");
        let result2 = handle2.join().expect("thread 2 panicked");

        assert!(result1.is_ok(), "peer 1 failed: {:?}", result1.err());
        assert!(result2.is_ok(), "peer 2 failed: {:?}", result2.err());
    }
}
