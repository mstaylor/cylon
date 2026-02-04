// Tests for SIMD operations

use cylon_wasm::simd::*;

#[test]
fn test_sum_f32() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let result = simd_sum_f32(&data);
    assert!((result - 45.0).abs() < 1e-6);
}

#[test]
fn test_sum_f32_empty() {
    let data: Vec<f32> = vec![];
    let result = simd_sum_f32(&data);
    assert!((result - 0.0).abs() < 1e-6);
}

#[test]
fn test_sum_f32_remainder() {
    // Non-multiple-of-4 length tests remainder handling
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let result = simd_sum_f32(&data);
    assert!((result - 15.0).abs() < 1e-6);
}

#[test]
fn test_sum_f64() {
    let data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
    let result = simd_sum_f64(&data);
    assert!((result - 15.0).abs() < 1e-10);
}

#[test]
fn test_dot_product_f32() {
    let a = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = vec![4.0f32, 3.0, 2.0, 1.0];
    let result = simd_dot_product_f32(&a, &b);
    // 1*4 + 2*3 + 3*2 + 4*1 = 20
    assert!((result - 20.0).abs() < 1e-6);
}

#[test]
fn test_dot_product_f64() {
    let a = vec![1.0f64, 2.0, 3.0];
    let b = vec![4.0f64, 5.0, 6.0];
    let result = simd_dot_product_f64(&a, &b);
    // 1*4 + 2*5 + 3*6 = 32
    assert!((result - 32.0).abs() < 1e-10);
}

#[test]
fn test_cosine_similarity_orthogonal() {
    let a = vec![1.0f32, 0.0, 0.0];
    let b = vec![0.0f32, 1.0, 0.0];
    let result = simd_cosine_similarity_f32(&a, &b);
    assert!(result.abs() < 1e-6);
}

#[test]
fn test_cosine_similarity_identical() {
    let a = vec![1.0f32, 2.0, 3.0];
    let b = vec![1.0f32, 2.0, 3.0];
    let result = simd_cosine_similarity_f32(&a, &b);
    assert!((result - 1.0).abs() < 1e-6);
}

#[test]
fn test_cosine_similarity_opposite() {
    let a = vec![1.0f32, 0.0, 0.0];
    let b = vec![-1.0f32, 0.0, 0.0];
    let result = simd_cosine_similarity_f32(&a, &b);
    assert!((result + 1.0).abs() < 1e-6);
}

#[test]
fn test_euclidean_distance() {
    let a = vec![0.0f32, 0.0, 0.0];
    let b = vec![3.0f32, 4.0, 0.0];
    let result = simd_euclidean_distance_f32(&a, &b);
    assert!((result - 5.0).abs() < 1e-6);
}

#[test]
fn test_euclidean_distance_same() {
    let a = vec![1.0f32, 2.0, 3.0];
    let b = vec![1.0f32, 2.0, 3.0];
    let result = simd_euclidean_distance_f32(&a, &b);
    assert!(result.abs() < 1e-6);
}

#[test]
fn test_min_f32() {
    let data = vec![5.0f32, 2.0, 8.0, 1.0, 9.0, 3.0];
    let result = simd_min_f32(&data);
    assert_eq!(result, Some(1.0));
}

#[test]
fn test_min_f32_empty() {
    let data: Vec<f32> = vec![];
    let result = simd_min_f32(&data);
    assert_eq!(result, None);
}

#[test]
fn test_max_f32() {
    let data = vec![5.0f32, 2.0, 8.0, 1.0, 9.0, 3.0];
    let result = simd_max_f32(&data);
    assert_eq!(result, Some(9.0));
}

#[test]
fn test_max_f32_negative() {
    let data = vec![-5.0f32, -2.0, -8.0, -1.0];
    let result = simd_max_f32(&data);
    assert_eq!(result, Some(-1.0));
}
