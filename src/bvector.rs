use std::ops;

use nalgebra::{SimdRealField, SVector};

pub trait LinAlgVector<T: SimdRealField> {
    fn dot(&self, other: &Self) -> T;
    fn len(&self) -> usize;
}

#[derive(Debug, Clone, PartialEq)]
pub struct BVector<T: SimdRealField, const N: usize> {
    pub data_vector: SVector<T, N>,
}

impl<T: SimdRealField, const N: usize> BVector<T, N> {
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
}

impl<T: SimdRealField, const N: usize> LinAlgVector<T> for BVector<T, N> {
    fn dot(&self, other: &Self) -> T {
        self.data_vector.dot(&other.data_vector)
    }

    fn len(&self) -> usize {
        N
    }
}

impl<T: SimdRealField, const N: usize> ops::Add for BVector<T, N> {
    type Output = BVector<T, N>;

    fn add(self, rhs: Self) -> Self::Output {
        BVector {
            data_vector: self.data_vector + rhs.data_vector,
        }
    }
}

/// Macro to construct a `BVector` with compile-time dimension inferred from the
/// number of elements provided.
///
/// Example:
/// - `let v = barray![1.0, 2.0, 3.0]; // BVector<f64, 3>`
/// - `let v = barray![2.0, -1.5]; // BVector<f64, 2>`
#[macro_export]
macro_rules! bvector {
    ($($x:expr),+ $(,)?) => {
        {
            // Let the compiler infer both element type `T` and length `N`.
            $crate::BVector::from_array([$( $x ),+])
        }
    };
}

