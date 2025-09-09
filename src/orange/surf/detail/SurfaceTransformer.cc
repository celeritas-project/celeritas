//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/detail/SurfaceTransformer.cc
//---------------------------------------------------------------------------//
#include "SurfaceTransformer.hh"

#include "corecel/cont/Range.hh"
#include "corecel/math/ArrayOperators.hh"
#include "corecel/math/ArrayUtils.hh"
#include "orange/MatrixUtils.hh"

#include "../ConeAligned.hh"
#include "../CylAligned.hh"
#include "../CylCentered.hh"
#include "../GeneralQuadric.hh"
#include "../Involute.hh"
#include "../Plane.hh"
#include "../PlaneAligned.hh"
#include "../SimpleQuadric.hh"
#include "../Sphere.hh"
#include "../SphereCentered.hh"

namespace celeritas
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
#define ORANGE_INSTANTIATE_OP(OUT, IN)                                     \
    template OUT SurfaceTransformer::operator()(IN<Axis::x> const&) const; \
    template OUT SurfaceTransformer::operator()(IN<Axis::y> const&) const; \
    template OUT SurfaceTransformer::operator()(IN<Axis::z> const&) const

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Transform an axis-aligned plane.
 */
template<Axis T>
Plane SurfaceTransformer::operator()(PlaneAligned<T> const& other) const
{
    return (*this)(Plane{other});
}

//! \cond
ORANGE_INSTANTIATE_OP(Plane, PlaneAligned);
//! \endcond

//---------------------------------------------------------------------------//
/*!
 * Transform a centered cylinder.
 */
template<Axis T>
GeneralQuadric SurfaceTransformer::operator()(CylCentered<T> const& other) const
{
    return (*this)(CylAligned<T>{other});
}

//! \cond
ORANGE_INSTANTIATE_OP(GeneralQuadric, CylCentered);
//! \endcond

//---------------------------------------------------------------------------//
/*!
 * Transform a sphere.
 */
Sphere SurfaceTransformer::operator()(SphereCentered const& other) const
{
    return (*this)(Sphere{other});
}

//---------------------------------------------------------------------------//
/*!
 * Transform an axis-aligned cylinder.
 */
template<Axis T>
GeneralQuadric SurfaceTransformer::operator()(CylAligned<T> const& other) const
{
    return (*this)(SimpleQuadric{other});
}

//! \cond
ORANGE_INSTANTIATE_OP(GeneralQuadric, CylAligned);
//! \endcond

//---------------------------------------------------------------------------//
/*!
 * Transform a plane.
 */
Plane SurfaceTransformer::operator()(Plane const& other) const
{
    // Rotate the normal direction
    Real3 normal = make_unit_vector(tr_.rotate_up(other.normal()));

    // Transform a point on the original plane
    Real3 point = tr_.transform_up(other.displacement() * other.normal());

    return Plane{normal, point};
}

//---------------------------------------------------------------------------//
/*!
 * Transform a sphere.
 */
Sphere SurfaceTransformer::operator()(Sphere const& other) const
{
    if (CELER_UNLIKELY(tr_.calc_properties().scales))
    {
        // Future work: return type needs to be gq
        CELER_NOT_IMPLEMENTED("transforming a sphere with scaling");
    }

    // Transform origin, keep the same radius
    return Sphere::from_radius_sq(tr_.transform_up(other.origin()),
                                  other.radius_sq());
}

//---------------------------------------------------------------------------//
/*!
 * Transform a cone.
 */
template<Axis T>
GeneralQuadric SurfaceTransformer::operator()(ConeAligned<T> const& other) const
{
    return (*this)(SimpleQuadric{other});
}

//! \cond
ORANGE_INSTANTIATE_OP(GeneralQuadric, ConeAligned);
//! \endcond

//---------------------------------------------------------------------------//
/*!
 * Transform a simple quadric.
 */
GeneralQuadric SurfaceTransformer::operator()(SimpleQuadric const& other) const
{
    return (*this)(GeneralQuadric{other});
}

//---------------------------------------------------------------------------//
/*!
 * Transform a quadric.
 *
 * See celeritas-doc/nb/geometry/quadric-transform.ipynb . The implementation
 * below is less than optimal because we don't need to explicitly construct the
 * full Q matrix.
 *
 * The inverse transform is: \f[
    \mathbf{R}^{-1}(\mathbf{x}' - \mathbf{t}) \to \mathbf{x}
    \f]
 *  or
    \f[
    \mathbf{\tilde x}
    \gets
    \begin{bmatrix}
    1 & 0 \\
    -\mathbf{R}^{-1} \mathbf{t} & \mathbf{R}^{-1} \\
    \end{bmatrix}
    \mathbf{\tilde x}'
    = \mathbf{\tilde R}^{-1} \mathbf{\tilde x}'
    \f]
 *
 */
GeneralQuadric SurfaceTransformer::operator()(GeneralQuadric const& other) const
{
    using Vec4 = Array<real_type, 4>;
    using Mat4 = SquareMatrix<real_type, 4>;

    // Build inverse transform matrix
    Mat4 const tr_inv = [this] {
        // Reverse and rotate translation
        Real3 trans = tr_.rotate_down(Real3{0, 0, 0} - tr_.translation());

        // Combine inverted translation with inverse (transpose) of rotation
        Mat4 tr_inv;
        tr_inv[0][0] = 1;
        for (auto i : range(1, 4))
        {
            tr_inv[i][0] = trans[i - 1];
            tr_inv[0][i] = 0;
            for (auto j : range(1, 4))
            {
                tr_inv[i][j] = tr_.rotation()[j - 1][i - 1];
            }
        }
        return tr_inv;
    }();

    auto calc_q = [&other] {
        constexpr auto X = to_int(Axis::x);
        constexpr auto Y = to_int(Axis::y);
        constexpr auto Z = to_int(Axis::z);

        Real3 const second = make_array(other.second());
        Real3 const cross = make_array(other.cross()) / real_type(2);
        Real3 const first = make_array(other.first()) / real_type(2);
        real_type const zeroth = other.zeroth();

        return Mat4{Vec4{zeroth, first[X], first[Y], first[Z]},
                    Vec4{first[X], second[X], cross[X], cross[Z]},
                    Vec4{first[Y], cross[X], second[Y], cross[Y]},
                    Vec4{first[Z], cross[Z], cross[Y], second[Z]}};
    };

    // Apply original quadric matrix to inverse transforom
    auto qrinv = gemm(calc_q(), tr_inv);

    // Apply transpose of inverse (result should be symmetric)
    auto qprime = gemm(matrix::transpose, tr_inv, qrinv);

    // Extract elements back from the matrix
    return GeneralQuadric(
        {qprime[1][1], qprime[2][2], qprime[3][3]},
        {2 * qprime[1][2], 2 * qprime[2][3], 2 * qprime[1][3]},
        {2 * qprime[0][1], 2 * qprime[0][2], 2 * qprime[0][3]},
        qprime[0][0]);
}

//---------------------------------------------------------------------------//
/*!
 * Transform an involute.
 */
Involute SurfaceTransformer::operator()(Involute const&) const
{
    CELER_NOT_IMPLEMENTED("transformed involutes");
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
