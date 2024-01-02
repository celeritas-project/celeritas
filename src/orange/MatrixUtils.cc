//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/MatrixUtils.cc
//---------------------------------------------------------------------------//
#include "MatrixUtils.hh"

#include <cmath>

#include "corecel/math/ArrayUtils.hh"
#include "corecel/math/SoftEqual.hh"

using Mat3 = celeritas::SquareMatrixReal3;

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Calculate the determiniant of a 3x3 matrix.
 */
template<class T>
T determinant(SquareMatrix<T, 3> const& mat)
{
    return mat[0][0] * mat[1][1] * mat[2][2] + mat[1][0] * mat[2][1] * mat[0][2]
           + mat[2][0] * mat[0][1] * mat[1][2]
           - mat[2][0] * mat[1][1] * mat[0][2]
           - mat[1][0] * mat[0][1] * mat[2][2]
           - mat[0][0] * mat[2][1] * mat[1][2];
}

//---------------------------------------------------------------------------//
/*!
 * Naive square matrix-matrix multiply.
 *
 * \f[
 * C \gets A * B
 * \f]
 *
 * This should be equivalent to BLAS' GEMM without the option to transpose,
 * use strides, or multiply by constants. All matrix orderings are C-style:
 * mat[i][j] is for row i, column j .
 *
 * \warning This implementation is limited and slow.
 */
template<class T, size_type N>
SquareMatrix<T, N>
gemm(SquareMatrix<T, N> const& a, SquareMatrix<T, N> const& b)
{
    SquareMatrix<T, N> result;
    for (size_type i = 0; i != N; ++i)
    {
        for (size_type j = 0; j != N; ++j)
        {
            // Reset target row
            result[i][j] = 0;
            // Accumulate dot products
            for (size_type k = 0; k != N; ++k)
            {
                result[i][j] = std::fma(b[k][j], a[i][k], result[i][j]);
            }
        }
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Naive square matrix-matrix multiply with the first matrix transposed.
 *
 * \f[
 * C \gets A^T * B
 * \f]
 *
 * \note The first argument is a "tag" that alters the behavior of this
 * function versus the non-transposed one.
 */
template<class T, size_type N>
SquareMatrix<T, N> gemm(matrix::TransposePolicy,
                        SquareMatrix<T, N> const& a,
                        SquareMatrix<T, N> const& b)
{
    SquareMatrix<T, N> result;
    for (size_type i = 0; i != N; ++i)
    {
        for (size_type j = 0; j != N; ++j)
        {
            // Reset target row
            result[i][j] = 0;
            // Accumulate dot products
            for (size_type k = 0; k != N; ++k)
            {
                result[i][j] = std::fma(b[k][j], a[k][i], result[i][j]);
            }
        }
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Normalize and orthogonalize a small, dense matrix.
 *
 * This is used for constructing rotation matrices from user-given matrices
 * that may only have a few digits of precision (e.g. were read from an XML
 * file). It uses the modified Gram-Schmidt orthogonalization algorithm.
 *
 * If debug assertions are enabled, the normality of the resulting matrix will
 * be checked. A singular matrix will fail.
 */
template<class T, size_type N>
void orthonormalize(SquareMatrix<T, N>* mat)
{
    for (size_type i = 0; i != N; ++i)
    {
        Array<T, N>& cur = (*mat)[i];

        // Orthogonalize against previous rows
        for (size_type ip = 0; ip != i; ++ip)
        {
            Array<T, N>& prev = (*mat)[ip];
            T proj = dot_product(cur, prev);
            axpy(-proj, prev, &cur);
        }

        // Normalize row
        T inv_mag = 1 / norm(cur);
        for (size_type j = 0; j != N; ++j)
        {
            cur[j] *= inv_mag;
        }
    }

    // Check result for orthonormality
    CELER_ENSURE(soft_equal(std::fabs(determinant(*mat)), T{1}));
}

//---------------------------------------------------------------------------//
/*!
 * Create a C-ordered rotation matrix.
 */
Mat3 make_rotation(Axis ax, Turn theta)
{
    CELER_EXPECT(ax < Axis::size_);

    // Calculate sin and cosine with less precision loss using "turn" value
    real_type cost;
    real_type sint;
    sincos(theta, &sint, &cost);

    // Fill result with zeros
    Mat3 r;
    r.fill(Real3{0, 0, 0});

    // {i, i} gets 1
    r[to_int(ax)][to_int(ax)] = 1;

    int uax = (to_int(ax) + 1) % 3;
    int vax = (to_int(ax) + 2) % 3;
    r[uax][uax] = cost;
    r[uax][vax] = negate(sint);  // avoid signed zeros
    r[vax][uax] = sint;
    r[vax][vax] = cost;
    return r;
}

//---------------------------------------------------------------------------//
/*!
 * Rotate a C-ordered rotation matrix.
 *
 * This applies the new axis + turn as a rotation operator to the left of
 * the matrix.
 *
 * For example, to rotate first by 135 degrees about the Z axis, then 90
 * degrees about the X axis:
 * \code
   auto r = make_rotation(Axes::x, Turn{0.25}, make_rotation(Axes::z, 0.375));
 * \endcode
 */
Mat3 make_rotation(Axis ax, Turn theta, Mat3 const& other)
{
    return gemm(make_rotation(ax, theta), other);
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATIONS
//---------------------------------------------------------------------------//
template int determinant(SquareMatrix<int, 3> const&);
template float determinant(SquareMatrix<float, 3> const&);
template double determinant(SquareMatrix<double, 3> const&);

template void orthonormalize(SquareMatrix<float, 3>*);
template void orthonormalize(SquareMatrix<double, 3>*);

// GEMM
template SquareMatrix<int, 3>
gemm(SquareMatrix<int, 3> const&, SquareMatrix<int, 3> const&);
template SquareMatrix<float, 3>
gemm(SquareMatrix<float, 3> const&, SquareMatrix<float, 3> const&);
template SquareMatrix<double, 3>
gemm(SquareMatrix<double, 3> const&, SquareMatrix<double, 3> const&);

// GEMM transpose
template SquareMatrix<int, 3> gemm(matrix::TransposePolicy,
                                   SquareMatrix<int, 3> const&,
                                   SquareMatrix<int, 3> const&);
template SquareMatrix<float, 3> gemm(matrix::TransposePolicy,
                                     SquareMatrix<float, 3> const&,
                                     SquareMatrix<float, 3> const&);
template SquareMatrix<double, 3> gemm(matrix::TransposePolicy,
                                      SquareMatrix<double, 3> const&,
                                      SquareMatrix<double, 3> const&);

// 4x4 real GEMM and transpose
template SquareMatrix<real_type, 4>
gemm(SquareMatrix<real_type, 4> const&, SquareMatrix<real_type, 4> const&);
template SquareMatrix<real_type, 4> gemm(matrix::TransposePolicy,
                                         SquareMatrix<real_type, 4> const&,
                                         SquareMatrix<real_type, 4> const&);

//---------------------------------------------------------------------------//
}  // namespace celeritas
