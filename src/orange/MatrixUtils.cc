//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/MatrixUtils.cc
//---------------------------------------------------------------------------//
#include "MatrixUtils.hh"

#include <algorithm>
#include <cmath>

#include "corecel/cont/Range.hh"
#include "corecel/math/Algorithms.hh"
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
    // clang-format off
    return   mat[0][0] * mat[1][1] * mat[2][2]
           + mat[1][0] * mat[2][1] * mat[0][2]
           + mat[2][0] * mat[0][1] * mat[1][2]
           - mat[2][0] * mat[1][1] * mat[0][2]
           - mat[1][0] * mat[0][1] * mat[2][2]
           - mat[0][0] * mat[2][1] * mat[1][2];
    // clang-format on
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the trace of a 3x3 matrix.
 *
 * The trace is just the sum of the diagonal elements.
 */
template<class T>
T trace(SquareMatrix<T, 3> const& mat)
{
    return mat[0][0] + mat[1][1] + mat[2][2];
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
 * Note that this uses \c celeritas::fma which supports types other than
 * floating point.
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
                result[i][j] = fma(b[k][j], a[i][k], result[i][j]);
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
                result[i][j] = fma(b[k][j], a[k][i], result[i][j]);
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
 * Create a C-ordered rotation matrix from an arbitrary rotation.
 *
 * This is equation (38) in "Rotation Matrices in Two, Three, and Many
 * Dimensions",  Physics 116A, UC Santa Cruz,
 * http://scipp.ucsc.edu/~haber/ph116A/.
 *
 * \param ax Axis of rotation (unit vector)
 * \param theta Rotation
 */
Mat3 make_rotation(Real3 const& ax, Turn theta)
{
    CELER_EXPECT(is_soft_unit_vector(ax));
    CELER_EXPECT(theta >= Turn{0} && theta <= Turn{real_type(0.5)});

    // Axis/direction enumeration
    enum
    {
        X = 0,
        Y = 1,
        Z = 2
    };

    // Calculate sin and cosine with less precision loss using "turn" value
    real_type cost;
    real_type sint;
    sincos(theta, &sint, &cost);

    Mat3 r{Real3{cost + ipow<2>(ax[X]) * (1 - cost),
                 ax[X] * ax[Y] * (1 - cost) - ax[Z] * sint,
                 ax[X] * ax[Z] * (1 - cost) + ax[Y] * sint},
           Real3{ax[X] * ax[Y] * (1 - cost) + ax[Z] * sint,
                 cost + ipow<2>(ax[Y]) * (1 - cost),
                 ax[Y] * ax[Z] * (1 - cost) - ax[X] * sint},
           Real3{ax[X] * ax[Z] * (1 - cost) - ax[Y] * sint,
                 ax[Y] * ax[Z] * (1 - cost) + ax[X] * sint,
                 cost + ipow<2>(ax[Z]) * (1 - cost)}};
    CELER_ENSURE(soft_equal(std::fabs(determinant(r)), real_type{1}));
    return r;
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
 * For example, to rotate first by 135 degrees about the *z* axis, then 90
 * degrees about the *x* axis:
 * \code
   auto r = make_rotation(Axes::x, Turn{0.25}, make_rotation(Axes::z, 0.375));
 * \endcode
 */
Mat3 make_rotation(Axis ax, Turn theta, Mat3 const& other)
{
    return gemm(make_rotation(ax, theta), other);
}

//---------------------------------------------------------------------------//
/*!
 * Create an identity matrix.
 */
SquareMatrixReal3 make_identity()
{
    SquareMatrixReal3 result;
    result.fill(Real3{0, 0, 0});
    result[0][0] = result[1][1] = result[2][2] = 1;
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Create a uniform scaling matrix.
 *
 * This creates a matrix that scales by the given factor along all axes.
 *
 * \param scale Scale factor to apply
 */
SquareMatrixReal3 make_scaling(real_type scale)
{
    CELER_EXPECT(scale > 0);
    return make_scaling(Real3{scale, scale, scale});
}

//---------------------------------------------------------------------------//
/*!
 * Create a scaling matrix along a given axis.
 *
 * This creates a matrix that scales by the given factor along the specified
 * axis while preserving the other dimensions.
 *
 * \param ax Axis along which to scale
 * \param scale Scale factor to apply
 */
SquareMatrixReal3 make_scaling(Axis ax, real_type scale)
{
    CELER_EXPECT(ax < Axis::size_);
    CELER_EXPECT(scale > 0);

    Real3 temp_scale{1, 1, 1};
    temp_scale[to_int(ax)] = scale;
    return make_scaling(temp_scale);
}

//---------------------------------------------------------------------------//
/*!
 * Create a scaling matrix along all three Cartesian axes.
 *
 * \param scale Scale factor for each axis
 */
SquareMatrixReal3 make_scaling(Real3 const& scale)
{
    CELER_EXPECT(std::all_of(
        scale.begin(), scale.end(), [](real_type s) { return s > 0; }));

    SquareMatrixReal3 result = make_identity();

    // Apply scale factor to each axis
    for (auto ax : range(to_int(Axis::size_)))
    {
        result[ax][ax] = scale[ax];
    }

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Create a reflection matrix perpendicular to a given axis.
 *
 * This creates a matrix that reflects across a plane on the origin, normal to
 * the specified axis. The sign of the coordinate on that axis is reversed.
 *
 * \param ax Axis to reflect
 * \return Matrix that represents reflection across the given axis
 */
SquareMatrixReal3 make_reflection(Axis ax)
{
    CELER_EXPECT(ax < Axis::size_);

    SquareMatrixReal3 result = make_identity();

    // Reflect the given axis
    result[to_int(ax)][to_int(ax)] = -1;

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct a transposed matrix.
 *
 * This should only be used for preprocessing. Prefer methods that transpose on
 * the fly.
 */
SquareMatrixReal3 make_transpose(SquareMatrixReal3 const& mat)
{
    SquareMatrixReal3 result;
    for (size_type i = 0; i != 3; ++i)
    {
        for (size_type j = 0; j != 3; ++j)
        {
            result[i][j] = mat[j][i];
        }
    }

    return result;
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATIONS
//---------------------------------------------------------------------------//
template int determinant(SquareMatrix<int, 3> const&);
template float determinant(SquareMatrix<float, 3> const&);
template double determinant(SquareMatrix<double, 3> const&);

template int trace(SquareMatrix<int, 3> const&);
template float trace(SquareMatrix<float, 3> const&);
template double trace(SquareMatrix<double, 3> const&);

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
