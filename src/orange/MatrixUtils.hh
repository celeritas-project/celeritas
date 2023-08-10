//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/MatrixUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/cont/Array.hh"
#include "corecel/math/Turn.hh"

#include "Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// Apply a matrix to an array
template<class T, size_type N>
inline CELER_FUNCTION Array<T, N> gemv(T alpha,
                                       SquareMatrix<T, N> const& a,
                                       Array<T, N> const& x,
                                       T beta,
                                       Array<T, N> const& y);

//---------------------------------------------------------------------------//
//! Apply a matrix to an array without scaling or addition
template<class T, size_type N>
inline CELER_FUNCTION Array<T, N>
gemv(SquareMatrix<T, N> const& a, Array<T, N> const& x)
{
    return gemv(T{1}, a, x, T{0}, x);
}

//---------------------------------------------------------------------------//
// Host-only declarations
// (double and float (and some int) for N=3 are instantiated in MatrixUtils.cc)
//---------------------------------------------------------------------------//

// Calculate the determinant of a matrix
template<class T>
T determinant(SquareMatrix<T, 3> const& mat);

// Perform a matrix-matrix multiply
template<class T, size_type N>
SquareMatrix<T, N>
gemm(SquareMatrix<T, N> const& a, SquareMatrix<T, N> const& b);

// Normalize and orthogonalize a small, dense matrix
template<class T, size_type N>
void orthonormalize(SquareMatrix<T, N>* mat);

// Create a C-ordered rotation matrix
SquareMatrixReal3 make_rotation(Axis ax, Turn rev);

// Apply a rotation to an existing C-ordered rotation matrix
SquareMatrixReal3 make_rotation(Axis ax, Turn rev, SquareMatrixReal3 const&);

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
 * This should be equivalent to BLAS' GEMV without the option to transpose. All
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
            result[i] += alpha * (a[i][j] * x[j]);
        }
    }
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
