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
    int first{0};  // first nonzero sign

    //! Whether any nonzero values exist
    explicit operator bool() const { return pos || neg; }

    //! Whether there are more negatives than positives
    //! *or* the first nonzero sign is negative
    bool should_flip() const
    {
        return *this && (neg > pos || (neg == pos && first < 0));
    }
};

SignCount count_signs(Span<real_type const, 3> arr, real_type tol)
{
    SignCount result;
    for (auto v : arr)
    {
        if (std::fabs(v) < tol)
        {
            // Effectively zero
            continue;
        }

        if (v < 0)
        {
            ++result.neg;
        }
        else
        {
            ++result.pos;
        }

        if (result.first == 0)
        {
            result.first = v < 0 ? -1 : 1;
        }
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
 */
auto SurfaceSimplifier::operator()(Plane const& p) const
    -> Optional<PlaneX, PlaneY, PlaneZ, Plane>
{
    auto signs = count_signs(make_span(p.normal()), tol_);
    CELER_ASSERT(signs);

    if (signs.should_flip())
    {
        // Flip the sense and reverse the values so that there are more
        // positives than negatives, or so the positive comes first
        *sense_ = flip_sense(*sense_);
        return negate_coefficients(p);
    }
    else if (signs.pos == 1 && signs.neg == 0)
    {
        // Only one plane direction is positive: snap to axis-aligned
        Real3 const& n = p.normal();
        if (n[to_int(Axis::x)] > tol_)
        {
            return PlaneX{p.displacement()};
        }
        else if (n[to_int(Axis::y)] > tol_)
        {
            return PlaneY{p.displacement()};
        }
        else
        {
            CELER_ASSUME(n[to_int(Axis::z)] > tol_);
            return PlaneZ{p.displacement()};
        }
    }

    Real3 n{p.normal()};
    real_type d{p.displacement()};

    // Snap nearly-zero normals to zero
    std::transform(n.begin(), n.end(), n.begin(), ZeroSnapper{tol_});

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
 *
 * The sign can also be reversed as part of regularization.
 *
 * \note Differently scaled SQs are *not* simplified at the moment due to small
 * changes in the intercept distances that haven't yet been investigated.
 * Geant4's GenericTrap twisted surfaces *are* normalized by the magnitude of
 * their linear component.
 */
auto SurfaceSimplifier::operator()(SimpleQuadric const& sq) const
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

    if (!signs)
    {
        // It's a plane
        return detail::QuadricPlaneConverter{tol_}(sq);
    }
    else if (signs.should_flip())
    {
        // Flip the sense and reverse the values so that there are more
        // positives than negatives, or so the positive comes first
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
 * - When the higher-order terms are negative, the signs will be flipped.
 *
 * \note Differently scaled GQs are *not* simplified at the moment due to small
 * changes in the intercept distances that haven't yet been investigated.
 */
auto SurfaceSimplifier::operator()(GeneralQuadric const& gq) const
    -> Optional<SimpleQuadric, GeneralQuadric>
{
    // Cross term signs
    auto csigns = count_signs(gq.cross(), tol_);
    if (!csigns)
    {
        // No cross terms
        return SimpleQuadric{
            make_array(gq.second()), make_array(gq.first()), gq.zeroth()};
    }

    // Second-order term signs
    auto ssigns = count_signs(gq.second(), tol_);
    if (ssigns.should_flip() || (!ssigns && csigns.should_flip()))
    {
        // More negative signs than positive:
        // flip the sense and reverse the values
        *sense_ = flip_sense(*sense_);
        return negate_coefficients(gq);
    }

    // No simplification
    return {};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
