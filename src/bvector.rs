// The type is deprecated, but this module still defines and tests it.
#![allow(deprecated)]

use std::ops;

use nalgebra::{SimdRealField, SVector};

/// A vector with a compile-time dimension, once the input type of [`crate::Perceptron`].
///
/// Deprecated: [`crate::Perceptron`] now takes plain arrays, so `bvector![a, b]` becomes
/// `[a, b]`. An existing `BVector` converts with `.into()`.
#[deprecated(since = "0.5.0", note = "use a plain array `[T; N]`, which `Perceptron` now takes; convert an existing `BVector` with `.into()`")]
#[derive(Debug, Clone, PartialEq)]
pub struct BVector<T: SimdRealField + Copy, const N: usize> {
    data_vector: SVector<T, N>,
}

impl<T: SimdRealField + Copy, const N: usize> BVector<T, N> {
    /// Constructs a vector filled with `element` of a dimension `N`.
    pub fn from_element(element: T) -> Self {
        BVector {
            data_vector: SVector::<T, N>::from_element(element),
        }
    }

    /// Constructs the vector from fixed-size array.
    pub fn from_array(data: [T; N]) -> Self
    where
        T: Copy,
    {
        BVector {
            data_vector: SVector::<T, N>::from_row_slice(&data),
        }
    }

    pub fn dot(&self, other: &Self) -> T {
        self.data_vector.dot(&other.data_vector)
    }

    pub fn len(&self) -> usize {
        N
    }

    /// Whether the vector has no elements, which is only the case when `N` is 0.
    pub fn is_empty(&self) -> bool {
        N == 0
    }

    pub fn get(&self, index: usize) -> T {
        assert!(index < N, "Index out of bounds");
        self.data_vector[index]
    }
}

impl<T: SimdRealField + Copy, const N: usize> ops::Add for BVector<T, N> {
    type Output = BVector<T, N>;

    fn add(self, rhs: Self) -> Self::Output {
        BVector {
            data_vector: self.data_vector + rhs.data_vector,
        }
    }
}

impl<T: SimdRealField + Copy, const N: usize> From<BVector<T, N>> for [T; N] {
    /// The elements in order, for code moving from `BVector` to plain arrays.
    fn from(vector: BVector<T, N>) -> Self {
        std::array::from_fn(|index| vector.data_vector[index])
    }
}

/// Macro to construct a `BVector` with compile-time dimension inferred from the
/// number of elements provided.
///
/// Deprecated with [`BVector`]: write the array itself, `[1.0, 2.0, 3.0]`.
#[deprecated(since = "0.5.0", note = "use a plain array, `[a, b, c]`")]
#[macro_export]
macro_rules! bvector {
    ($($x:expr),+ $(,)?) => {
        {
            // Let the compiler infer both element type `T` and length `N`.
            $crate::BVector::from_array([$( $x ),+])
        }
    };
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_element_fills_every_position() {
        let v = BVector::<f64, 3>::from_element(1.5);

        assert_eq!((v.get(0), v.get(1), v.get(2)), (1.5, 1.5, 1.5));
    }

    #[test]
    fn the_macro_keeps_elements_in_order() {
        let v = bvector![1.0, 2.0, 3.0];

        assert_eq!(v, BVector::from_array([1.0, 2.0, 3.0]));
        assert_eq!((v.get(0), v.get(1), v.get(2)), (1.0, 2.0, 3.0));
    }

    #[test]
    fn the_macro_accepts_a_trailing_comma() {
        assert_eq!(bvector![1.0, 2.0,], bvector![1.0, 2.0]);
    }

    #[test]
    fn len_is_the_compile_time_dimension() {
        assert_eq!(bvector![1.0, 2.0, 3.0].len(), 3);
        assert!(!bvector![1.0].is_empty());
        assert!(BVector::<f64, 0>::from_array([]).is_empty());
    }

    #[test]
    fn dot_sums_the_element_wise_products() {
        assert_eq!(bvector![1.0, 2.0, 3.0].dot(&bvector![4.0, -5.0, 6.0]), 12.0);
    }

    #[test]
    fn add_is_element_wise() {
        assert_eq!(bvector![1.0, 2.0] + bvector![0.5, -3.0], bvector![1.5, -1.0]);
    }

    #[test]
    fn converts_into_an_array_in_order() {
        let array: [f64; 3] = bvector![1.0, -2.0, 3.0].into();

        assert_eq!(array, [1.0, -2.0, 3.0]);
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn get_rejects_an_index_past_the_end() {
        bvector![1.0, 2.0].get(2);
    }
}
