//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/MatrixUtils.hh
//! \todo Split into BLAS and host-only utils
//! \todo Investigate Eigen for setup-time computations if necessary
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/Turn.hh"
#include "geocel/Types.hh"

namespace celeritas
{
//! Policy tags for matrix operations
namespace matrix
{
//---------------------------------------------------------------------------//
struct TransposePolicy
{
};
//! Indicate that the input matrix is transposed
inline constexpr TransposePolicy transpose{};
}  // namespace matrix

//---------------------------------------------------------------------------//
// Apply a matrix to an array
template<class T, size_type N>
inline CELER_FUNCTION Array<T, N> gemv(T alpha,
                                       SquareMatrix<T, N> const& a,
                                       Array<T, N> const& x,
                                       T beta,
                                       Array<T, N> const& y);

//---------------------------------------------------------------------------//
// Apply the transpose of a matrix to an array
template<class T, size_type N>
inline CELER_FUNCTION Array<T, N> gemv(matrix::TransposePolicy,
                                       T alpha,
                                       SquareMatrix<T, N> const& a,
                                       Array<T, N> const& x,
                                       T beta,
                                       Array<T, N> const& y);

//---------------------------------------------------------------------------//
//!@{
//! Apply a matrix or its transpose to an array, without scaling or addition
template<class T, size_type N>
inline CELER_FUNCTION Array<T, N>
gemv(SquareMatrix<T, N> const& a, Array<T, N> const& x)
{
    return gemv(T{1}, a, x, T{0}, x);
}

template<class T, size_type N>
inline CELER_FUNCTION Array<T, N>
gemv(matrix::TransposePolicy, SquareMatrix<T, N> const& a, Array<T, N> const& x)
{
    return gemv(matrix::transpose, T{1}, a, x, T{0}, x);
}
//!@}
//---------------------------------------------------------------------------//
// Host-only declarations
// (double and float (and some int) for N=3 are instantiated in MatrixUtils.cc)
//---------------------------------------------------------------------------//

// Calculate the determinant of a matrix
template<class T>
T determinant(SquareMatrix<T, 3> const& mat);

// Calculate the trace of a matrix
template<class T>
T trace(SquareMatrix<T, 3> const& mat);

// Perform a matrix-matrix multiply
template<class T, size_type N>
SquareMatrix<T, N>
gemm(SquareMatrix<T, N> const& a, SquareMatrix<T, N> const& b);

// Perform a matrix-matrix multiply with A transposed
template<class T, size_type N>
SquareMatrix<T, N> gemm(matrix::TransposePolicy,
                        SquareMatrix<T, N> const& a,
                        SquareMatrix<T, N> const& b);

// Normalize and orthogonalize a small, dense matrix
template<class T, size_type N>
void orthonormalize(SquareMatrix<T, N>* mat);

// Create an identity 3x3 matrix
SquareMatrixReal3 make_identity();

// Create a C-ordered rotation matrix about an arbitrary axis
SquareMatrixReal3 make_rotation(Real3 const& ax, Turn rev);

// Create a C-ordered rotation matrix about a Cartesian axis
SquareMatrixReal3 make_rotation(Axis ax, Turn rev);

// Apply a rotation to an existing C-ordered rotation matrix
SquareMatrixReal3 make_rotation(Axis ax, Turn rev, SquareMatrixReal3 const&);

// Scale uniformly
SquareMatrixReal3 make_scaling(real_type scale);

// Scale along an axis
SquareMatrixReal3 make_scaling(Axis ax, real_type scale);

// Scale along all three Cartesian axes
SquareMatrixReal3 make_scaling(Real3 const& scale);

// Reflect across a plane perpendicular to the axis
SquareMatrixReal3 make_reflection(Axis ax);

// Construct a transposed matrix
SquareMatrixReal3 make_transpose(SquareMatrixReal3 const&);

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Naive generalized matrix-vector multiply.
 *
 * \f[
 * z \gets \alpha A x + \beta y
 * \f]
 *
 * This should be equivalent to BLAS' GEMV without transposition. All
 * matrix orderings are C-style: mat[i][j] is for row i, column j .
 *
 * \warning This implementation is limited and slow.
 */
template<class T, size_type N>
CELER_FUNCTION Array<T, N> gemv(T alpha,
                                SquareMatrix<T, N> const& a,
                                Array<T, N> const& x,
                                T beta,
                                Array<T, N> const& y)
{
    Array<T, N> result;
    for (size_type i = 0; i != N; ++i)
    {
        result[i] = beta * y[i];
        for (size_type j = 0; j != N; ++j)
        {
            result[i] = fma(alpha, a[i][j] * x[j], result[i]);
        }
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Naive transposed generalized matrix-vector multiply.
 *
 * \f[
 * z \gets \alpha A^T x + \beta y
 * \f]
 *
 * This should be equivalent to BLAS' GEMV with the 't' option. All
 * matrix orderings are C-style: mat[i][j] is for row i, column j .
 *
 * \warning This implementation is limited and slow.
 */
template<class T, size_type N>
CELER_FUNCTION Array<T, N> gemv(matrix::TransposePolicy,
                                T alpha,
                                SquareMatrix<T, N> const& a,
                                Array<T, N> const& x,
                                T beta,
                                Array<T, N> const& y)
{
    Array<T, N> result;
    for (size_type i = 0; i != N; ++i)
    {
        result[i] = beta * y[i];
    }
    for (size_type j = 0; j != N; ++j)
    {
        for (size_type i = 0; i != N; ++i)
        {
            result[i] = fma(alpha, a[j][i] * x[j], result[i]);
        }
    }
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
