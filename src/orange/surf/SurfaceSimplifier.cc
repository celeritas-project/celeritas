//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/SurfaceSimplifier.cc
//---------------------------------------------------------------------------//
#include "SurfaceSimplifier.hh"

#include "corecel/cont/Range.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayOperators.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/math/SoftEqual.hh"

#include "ConeAligned.hh"
#include "CylAligned.hh"
#include "CylCentered.hh"
#include "GeneralQuadric.hh"
#include "Plane.hh"
#include "PlaneAligned.hh"
#include "SimpleQuadric.hh"
#include "Sphere.hh"
#include "SphereCentered.hh"
#include "detail/PlaneAlignedConverter.hh"
#include "detail/QuadricConeConverter.hh"
#include "detail/QuadricCylConverter.hh"
#include "detail/QuadricPlaneConverter.hh"
#include "detail/QuadricSphereConverter.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
#define ORANGE_INSTANTIATE_OP(OUT, IN)                       \
    template SurfaceSimplifier::Optional<OUT<Axis::x>>       \
    SurfaceSimplifier::operator()(IN<Axis::x> const&) const; \
    template SurfaceSimplifier::Optional<OUT<Axis::y>>       \
    SurfaceSimplifier::operator()(IN<Axis::y> const&) const; \
    template SurfaceSimplifier::Optional<OUT<Axis::z>>       \
    SurfaceSimplifier::operator()(IN<Axis::z> const&) const

class ZeroSnapper
{
  public:
    explicit ZeroSnapper(real_type tol) : soft_zero_{tol} {}

    //! Transform the value so that near-zeros (and signed zeros) become zero
    real_type operator()(real_type v) const
    {
        if (soft_zero_(v))
            return 0;
        return v;
    }

  private:
    SoftZero<> soft_zero_;
};

struct SignCount
{
    int pos{0};
    int neg{0};
};

SignCount count_signs(Span<real_type const, 3> arr, real_type tol)
{
    SignCount result;
    for (auto v : arr)
    {
        if (v < -tol)
            ++result.neg;
        else if (v > tol)
            ++result.pos;
    }
    return result;
}

template<class S>
S negate_coefficients(S const& orig)
{
    auto arr = make_array(orig.data());
    for (real_type& v : arr)
    {
        v = negate(v);
    }

    // TODO: make_span doesn't use the correct overload and creates a
    // dynamic extent span
    using SpanT = decltype(orig.data());
    return S{SpanT{arr.data(), arr.size()}};
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Plane may be snapped to origin.
 */
template<Axis T>
auto SurfaceSimplifier::operator()(PlaneAligned<T> const& p) const
    -> Optional<PlaneAligned<T>>
{
    if (p.position() != real_type{0} && SoftZero{tol_}(p.position()))
    {
        // Snap to zero since it's not already zero
        return PlaneAligned<T>{real_type{0}};
    }
    // No simplification performed
    return {};
}
//! \cond
ORANGE_INSTANTIATE_OP(PlaneAligned, PlaneAligned);
//! \endcond

//---------------------------------------------------------------------------//
/*!
 * Cylinder at origin will be simplified.
 *
 * \verbatim
   distance({0,0}, {u,v}) < tol
   sqrt(u^2 + v^2) < tol
   u^2 + v^2 < tol^2
   \endverbatim
 */
template<Axis T>
auto SurfaceSimplifier::operator()(CylAligned<T> const& c) const
    -> Optional<CylCentered<T>>
{
    real_type origin_dist = ipow<2>(c.origin_u()) + ipow<2>(c.origin_v());
    if (origin_dist < ipow<2>(tol_))
    {
        // Snap to zero since it's not already zero
        return CylCentered<T>::from_radius_sq(c.radius_sq());
    }
    // No simplification performed
    return {};
}

//! \cond
ORANGE_INSTANTIATE_OP(CylCentered, CylAligned);
//! \endcond

//---------------------------------------------------------------------------//
/*!
 * A cone whose origin is close to any axis will be snapped to it.
 *
 * This uses a 1-norm for simplicity.
 */
template<Axis T>
auto SurfaceSimplifier::operator()(ConeAligned<T> const& c) const
    -> Optional<ConeAligned<T>>
{
    bool simplified = false;
    Real3 origin = c.origin();
    SoftZero const soft_zero{tol_};
    for (auto ax : range(to_int(Axis::size_)))
    {
        if (origin[ax] != 0 && soft_zero(origin[ax]))
        {
            origin[ax] = 0;
            simplified = true;
        }
    }

    if (simplified)
    {
        return ConeAligned<T>::from_tangent_sq(origin, c.tangent_sq());
    }

    // No simplification performed
    return {};
}

//! \cond
ORANGE_INSTANTIATE_OP(ConeAligned, ConeAligned);
//! \endcond

//---------------------------------------------------------------------------//
/*!
 * Plane may be flipped, adjusted, or become axis-aligned.
 *
 * If a plane has a normal of {-1, 0 + eps, 0}, it will first be truncated to
 * {-1, 0, 0}, then flipped to {1, 0, 0}, and a new Plane will be returned.
 * That plane can *then* be simplified to an axis-aligned one.
 */
auto SurfaceSimplifier::operator()(Plane const& p)
    -> Optional<PlaneX, PlaneY, PlaneZ, Plane>
{
    {
        // First, try to snap to aligned plane
        detail::PlaneAlignedConverter to_aligned{tol_};
        if (auto pa = to_aligned(AxisTag<Axis::x>{}, p))
            return *pa;
        if (auto pa = to_aligned(AxisTag<Axis::y>{}, p))
            return *pa;
        if (auto pa = to_aligned(AxisTag<Axis::z>{}, p))
            return *pa;
    }

    Real3 n{p.normal()};
    real_type d{p.displacement()};

    // Snap nearly-zero normals to zero
    std::transform(n.begin(), n.end(), n.begin(), ZeroSnapper{tol_});

    // To prevent opposite-value planes from being defined but not
    // deduplicated, ensure the first non-zero normal component is in the
    // positive half-space. This also takes care of flipping orthogonal planes
    // defined like {-x = 3}, translating them to { x = -3 }.
    for (auto ax : range(to_int(Axis::size_)))
    {
        if (n[ax] > 0)
        {
            break;
        }
        else if (n[ax] < 0)
        {
            // Flip the sign of this and any remaining nonzero axes
            // (previous axes are zero so just skip them)
            for (auto ax2 : range(ax, to_int(Axis::size_)))
            {
                n[ax2] = negate(n[ax2]);
            }
            // Flip sign of d (without introducing -0)
            d = negate(d);
            // Flip sense
            *sense_ = flip_sense(*sense_);
            break;
        }
    }

    if (n != p.normal())
    {
        // The direction was changed: renormalize and return the updated plane
        real_type norm_factor = 1 / celeritas::norm(n);
        n *= norm_factor;
        d *= norm_factor;
        return Plane{n, d};
    }

    if (d != 0 && SoftZero<>{tol_}(d))
    {
        // Snap zero-distances to zero
        return Plane{n, 0};
    }

    // No simplification performed
    return {};
}

//---------------------------------------------------------------------------//
/*!
 * Sphere near center can be snapped.
 */
auto SurfaceSimplifier::operator()(Sphere const& s) const
    -> Optional<SphereCentered>
{
    if (dot_product(s.origin(), s.origin()) < ipow<2>(tol_))
    {
        // Sphere is less than tolerance from the origin
        return SphereCentered::from_radius_sq(s.radius_sq());
    }
    // No simplification performed
    return {};
}

//---------------------------------------------------------------------------//
/*!
 * Simple quadric with near-zero terms can be another second-order surface.
 */
auto SurfaceSimplifier::operator()(SimpleQuadric const& sq)
    -> Optional<Plane,
                Sphere,
                CylAligned<Axis::x>,
                CylAligned<Axis::y>,
                CylAligned<Axis::z>,
                ConeAligned<Axis::x>,
                ConeAligned<Axis::y>,
                ConeAligned<Axis::z>,
                SimpleQuadric>
{
    // Determine possible simplifications by calculating number of zeros
    auto signs = count_signs(sq.second(), tol_);

    if (signs.pos == 0 && signs.neg == 0)
    {
        // It's a plane
        return detail::QuadricPlaneConverter{tol_}(sq);
    }
    else if (signs.neg > signs.pos)
    {
        // More positive signs than negative:
        // flip the sense and reverse the values
        *sense_ = flip_sense(*sense_);
        return negate_coefficients(sq);
    }
    else if (signs.pos == 3)
    {
        // Could be a sphere
        detail::QuadricSphereConverter to_sphere{tol_};
        if (auto s = to_sphere(sq))
            return *s;
    }
    else if (signs.pos == 2 && signs.neg == 1)
    {
        // Cone: one second-order term less than zero, others equal
        detail::QuadricConeConverter to_cone{tol_};
        if (auto c = to_cone(AxisTag<Axis::x>{}, sq))
            return *c;
        if (auto c = to_cone(AxisTag<Axis::y>{}, sq))
            return *c;
        if (auto c = to_cone(AxisTag<Axis::z>{}, sq))
            return *c;
    }
    else if (signs.pos == 2 && signs.neg == 0)
    {
        // Cyl: one second-order term is zero, others are equal
        detail::QuadricCylConverter to_cyl{tol_};
        if (auto c = to_cyl(AxisTag<Axis::x>{}, sq))
            return *c;
        if (auto c = to_cyl(AxisTag<Axis::y>{}, sq))
            return *c;
        if (auto c = to_cyl(AxisTag<Axis::z>{}, sq))
            return *c;
    }

    // No simplification performed
    return {};
}

//---------------------------------------------------------------------------//
/*!
 * Quadric can be regularized or simplified.
 *
 * - When no cross terms are present, it's "simple".
 * - When the higher-order terms are negative, the signs need to be flipped.
 */
auto SurfaceSimplifier::operator()(GeneralQuadric const& gq)
    -> Optional<SimpleQuadric, GeneralQuadric>
{
    // Cross term signs
    auto csigns = count_signs(gq.cross(), tol_);
    if (csigns.pos == 0 && csigns.neg == 0)
    {
        // No cross terms
        return SimpleQuadric{
            make_array(gq.second()), make_array(gq.first()), gq.zeroth()};
    }

    // Second-order term signs
    auto ssigns = count_signs(gq.second(), tol_);
    if (ssigns.neg > ssigns.pos
        || (ssigns.neg == 0 && ssigns.pos == 0 && csigns.neg > csigns.pos))
    {
        // More positive signs than negative:
        // flip the sense and reverse the values
        *sense_ = flip_sense(*sense_);
        return negate_coefficients(gq);
    }

    // No simplification
    return {};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
