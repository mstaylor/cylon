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

//! Tests for Libfabric configuration types
//!
//! These tests verify the configuration structs and their builder patterns.
//! They do not require libfabric runtime to be available.

#[cfg(feature = "libfabric")]
mod libfabric_config_tests {
    use cylon::net::libfabric::{
        LibfabricConfig, EndpointType, ProgressMode, OOBType,
    };
    use cylon::net::{CommConfig, CommType};

    #[test]
    fn test_endpoint_type_default() {
        let ep_type = EndpointType::default();
        assert_eq!(ep_type, EndpointType::ReliableDatagram);
    }

    #[test]
    fn test_progress_mode_default() {
        let mode = ProgressMode::default();
        assert_eq!(mode, ProgressMode::Auto);
    }

    #[test]
    fn test_oob_type_default() {
        let oob = OOBType::default();
        assert_eq!(oob, OOBType::Redis);
    }

    #[test]
    fn test_libfabric_config_default() {
        let config = LibfabricConfig::default();
        assert!(config.provider.is_none());
        assert_eq!(config.endpoint_type, EndpointType::ReliableDatagram);
        assert_eq!(config.cq_size, 1024);
        assert_eq!(config.av_size, 256);
        assert_eq!(config.progress_mode, ProgressMode::Auto);
        assert_eq!(config.oob_type, OOBType::Redis);
        assert_eq!(config.redis_host, "127.0.0.1");
        assert_eq!(config.redis_port, 6379);
        assert_eq!(config.world_size, 0);
        assert_eq!(config.max_eager_size, 8192);
        assert_eq!(config.timeout_ms, 30000);
    }

    #[test]
    fn test_libfabric_config_with_redis() {
        let config = LibfabricConfig::with_redis("10.0.0.1", 6380, "test-session", 4);
        assert_eq!(config.redis_host, "10.0.0.1");
        assert_eq!(config.redis_port, 6380);
        assert_eq!(config.session_id, "test-session");
        assert_eq!(config.world_size, 4);
        assert_eq!(config.oob_type, OOBType::Redis);
    }

    #[test]
    fn test_libfabric_config_with_provider() {
        let config = LibfabricConfig::with_provider("tcp");
        assert_eq!(config.provider, Some("tcp".to_string()));
    }

    #[test]
    fn test_libfabric_config_builder_pattern() {
        let config = LibfabricConfig::default()
            .provider("efa")
            .endpoint_type(EndpointType::Message)
            .cq_size(2048)
            .av_size(512)
            .redis("192.168.1.1", 6381)
            .session_id("my-session")
            .world_size(8)
            .timeout_ms(60000);

        assert_eq!(config.provider, Some("efa".to_string()));
        assert_eq!(config.endpoint_type, EndpointType::Message);
        assert_eq!(config.cq_size, 2048);
        assert_eq!(config.av_size, 512);
        assert_eq!(config.redis_host, "192.168.1.1");
        assert_eq!(config.redis_port, 6381);
        assert_eq!(config.session_id, "my-session");
        assert_eq!(config.world_size, 8);
        assert_eq!(config.timeout_ms, 60000);
    }

    #[test]
    fn test_libfabric_config_comm_type() {
        let config = LibfabricConfig::default();
        assert_eq!(config.get_type(), CommType::Libfabric);
    }

    #[test]
    fn test_libfabric_config_as_any() {
        let config = LibfabricConfig::default();
        let any_ref = config.as_any();
        assert!(any_ref.downcast_ref::<LibfabricConfig>().is_some());
    }

    #[test]
    fn test_endpoint_type_equality() {
        assert_eq!(EndpointType::ReliableDatagram, EndpointType::ReliableDatagram);
        assert_eq!(EndpointType::Message, EndpointType::Message);
        assert_ne!(EndpointType::ReliableDatagram, EndpointType::Message);
    }

    #[test]
    fn test_progress_mode_equality() {
        assert_eq!(ProgressMode::Auto, ProgressMode::Auto);
        assert_eq!(ProgressMode::Manual, ProgressMode::Manual);
        assert_ne!(ProgressMode::Auto, ProgressMode::Manual);
    }

    #[test]
    fn test_oob_type_equality() {
        assert_eq!(OOBType::Redis, OOBType::Redis);
        assert_eq!(OOBType::Mpi, OOBType::Mpi);
        assert_ne!(OOBType::Redis, OOBType::Mpi);
    }

    #[test]
    fn test_config_clone() {
        let config1 = LibfabricConfig::default()
            .provider("tcp")
            .world_size(4);
        let config2 = config1.clone();

        assert_eq!(config1.provider, config2.provider);
        assert_eq!(config1.world_size, config2.world_size);
        assert_eq!(config1.session_id, config2.session_id);
    }

    #[test]
    fn test_endpoint_type_copy() {
        let ep1 = EndpointType::ReliableDatagram;
        let ep2 = ep1;
        assert_eq!(ep1, ep2);
    }

    #[test]
    fn test_progress_mode_copy() {
        let pm1 = ProgressMode::Manual;
        let pm2 = pm1;
        assert_eq!(pm1, pm2);
    }

    #[test]
    fn test_config_builder_chaining() {
        // Test that builder methods can be chained in any order
        let config = LibfabricConfig::default()
            .timeout_ms(5000)
            .provider("verbs")
            .world_size(16)
            .cq_size(4096);

        assert_eq!(config.timeout_ms, 5000);
        assert_eq!(config.provider, Some("verbs".to_string()));
        assert_eq!(config.world_size, 16);
        assert_eq!(config.cq_size, 4096);
    }

    #[test]
    fn test_endpoint_type_debug() {
        let ep = EndpointType::ReliableDatagram;
        let debug_str = format!("{:?}", ep);
        assert!(debug_str.contains("ReliableDatagram"));
    }

    #[test]
    fn test_progress_mode_debug() {
        let pm = ProgressMode::Auto;
        let debug_str = format!("{:?}", pm);
        assert!(debug_str.contains("Auto"));
    }

    #[test]
    fn test_config_debug() {
        let config = LibfabricConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("LibfabricConfig"));
        assert!(debug_str.contains("cq_size"));
    }
}

#[cfg(not(feature = "libfabric"))]
mod libfabric_disabled {
    #[test]
    fn libfabric_feature_not_enabled() {
        println!("Libfabric config tests skipped - feature not enabled");
    }
}
