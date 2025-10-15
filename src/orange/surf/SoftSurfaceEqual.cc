//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/SoftSurfaceEqual.cc
//---------------------------------------------------------------------------//
#include "SoftSurfaceEqual.hh"

#include <cmath>
#include <limits>

#include "detail/AllSurfaces.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
#define ORANGE_INSTANTIATE_OP(S)                                         \
    template bool SoftSurfaceEqual::operator()(S<Axis::x> const&,        \
                                               S<Axis::x> const&) const; \
    template bool SoftSurfaceEqual::operator()(S<Axis::y> const&,        \
                                               S<Axis::y> const&) const; \
    template bool SoftSurfaceEqual::operator()(S<Axis::z> const&,        \
                                               S<Axis::z> const&) const

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Compare two aligned planes for near equality.
 */
template<Axis T>
bool SoftSurfaceEqual::operator()(PlaneAligned<T> const& a,
                                  PlaneAligned<T> const& b) const
{
    return soft_eq_(a.position(), b.position());
}

//! \cond
ORANGE_INSTANTIATE_OP(PlaneAligned);
//! \endcond

//---------------------------------------------------------------------------//
/*!
 * Compare two centered axis-aligned cylinders for near equality.
 */
template<Axis T>
bool SoftSurfaceEqual::operator()(CylCentered<T> const& a,
                                  CylCentered<T> const& b) const
{
    return this->soft_eq_sq(a.radius_sq(), b.radius_sq());
}

//! \cond
ORANGE_INSTANTIATE_OP(CylCentered);
//! \endcond

//---------------------------------------------------------------------------//
/*!
 * Compare two centered spheres for near equality.
 */
bool SoftSurfaceEqual::operator()(SphereCentered const& a,
                                  SphereCentered const& b) const
{
    return this->soft_eq_sq(a.radius_sq(), b.radius_sq());
}

//---------------------------------------------------------------------------//
/*!
 * Compare two aligned cylinders for near equality.
 */
template<Axis T>
bool SoftSurfaceEqual::operator()(CylAligned<T> const& a,
                                  CylAligned<T> const& b) const
{
    return this->soft_eq_sq(a.radius_sq(), b.radius_sq())
           && this->soft_eq_distance(a.calc_origin(), b.calc_origin());
}

//! \cond
ORANGE_INSTANTIATE_OP(CylAligned);
//! \endcond

//---------------------------------------------------------------------------//
/*!
 * Compare two planes for near equality.
 *
 * Consider two planes with normals \em a and \em b , without loss of
 * generality where \em a is at \f$x=0\f$ and \em b is in the \em xy
 * plane. Suppose that to be equal, we want a intercept error of no more than
 * \f$\delta\f$ at a unit distance from the normal (since we're assuming the
 * error is based on the length scale of the problem). Taking a ray
 * from (1, 1) along (-1, 0), the distance to the plane with normal \em a is 1,
 * and the distance to plane \em b is then \f$ 1 \pm \delta \f$. This results
 * in a right triangle with legs of 1 and \f$ \delta \f$ and opening angle from
 * the origin of \f$ \theta \f$, which is equal to the angle between the
 * normals of the two
 * planes. Thus we have:
 * \f[
   \tan \theta = \frac{\delta}{1}
 * \f]
 * and
 * \f[
   \cos \theta = a \cdot b \equiv \mu \;.
 * \f]
 * thus \f[
   \mu = \frac{1}{\sqrt{1 + \delta^2}}
   \to
   \delta = \sqrt{\frac{1}{\mu^2} - 1}
 * \f]
 * so if we want to limit the intercept error to \f$ \epsilon \f$, then
 * \f[
   \sqrt{\frac{1}{\mu^2} - 1} < \epsilon \;.
 * \f]
 * and we also have to make sure the two planes are pointed into the same
 * half-space by checking for \f$ \mu > 0 \f$.
 *
 * Since this derivation is based on an absolute length scale of 1, the
 * relative tolerance should be used.
 *
 * Due to floating point arithmetic, \f$mu\f$ can be slightly greater than
 * unity, and since epsilon is often smaller than
 * \f$\sqrt{\epsilon_\mathrm{mach}}\f$ for single precision arithmetic, the
 * comparison here adds an extra bump to account for the precision loss.
 *
 * \image html orange-surface-softeq-plane.png
 *
 * \todo Just use SoftClose for each planar component?
 */
bool SoftSurfaceEqual::operator()(Plane const& a, Plane const& b) const
{
    if (!soft_eq_(a.displacement(), b.displacement()))
        return false;

    real_type const mu = dot_product(a.normal(), b.normal());
    constexpr real_type eps_mach = std::numeric_limits<real_type>::epsilon();
    return mu > 0
           && (1 / ipow<2>(mu) - 1) <= (ipow<2>(soft_eq_.rel()) + eps_mach);
}

//---------------------------------------------------------------------------//
/*!
 * Compare two spheres for near equality.
 */
bool SoftSurfaceEqual::operator()(Sphere const& a, Sphere const& b) const
{
    return this->soft_eq_sq(a.radius_sq(), b.radius_sq())
           && this->soft_eq_distance(a.origin(), b.origin());
}

//---------------------------------------------------------------------------//
/*!
 * Compare two cones for near equality.
 */
template<Axis T>
bool SoftSurfaceEqual::operator()(ConeAligned<T> const& a,
                                  ConeAligned<T> const& b) const
{
    return this->soft_eq_sq(a.tangent_sq(), b.tangent_sq())
           && this->soft_eq_distance(a.origin(), b.origin());
}

//! \cond
ORANGE_INSTANTIATE_OP(ConeAligned);
//! \endcond

//---------------------------------------------------------------------------//
/*!
 * Compare two simple quadrics for near equality.
 *
 * \note This is an ad hoc comparison. To compare correctly on zeroth order
 * terms, we need the quadrics to be scaled so that the constant is order \c
 * length not \c squared length. Then using a soft equivalence on each element
 * would more or less work?
 */
bool SoftSurfaceEqual::operator()(SimpleQuadric const& a,
                                  SimpleQuadric const& b) const
{
    return this->soft_eq_distance(make_array(a.second()),
                                  make_array(b.second()))
           && this->soft_eq_distance(make_array(a.first()),
                                     make_array(b.first()))
           && soft_eq_(a.zeroth(), b.zeroth());
}

//---------------------------------------------------------------------------//
/*!
 * Compare two general quadrics for near equality.
 *
 * \todo See note on SQ.
 */
bool SoftSurfaceEqual::operator()(GeneralQuadric const& a,
                                  GeneralQuadric const& b) const
{
    return this->soft_eq_distance(make_array(a.second()),
                                  make_array(b.second()))
           && this->soft_eq_distance(make_array(a.cross()),
                                     make_array(b.cross()))
           && this->soft_eq_distance(make_array(a.first()),
                                     make_array(b.first()))
           && soft_eq_(a.zeroth(), b.zeroth());
}

//---------------------------------------------------------------------------//
/*!
 * Compare two centered involutes for near equality.
 */
bool SoftSurfaceEqual::operator()(Involute const& a, Involute const& b) const
{
    return this->soft_eq_(a.r_b(), b.r_b())
           && this->soft_eq_(a.displacement_angle(), b.displacement_angle())
           && a.sign() == b.sign() && this->soft_eq_(a.tmin(), b.tmin())
           && this->soft_eq_(a.tmax(), b.tmax())
           && this->soft_eq_distance({a.origin()[0], a.origin()[1], 0},
                                     {b.origin()[0], b.origin()[1], 0});
}

//---------------------------------------------------------------------------//
// PRIVATE HELPER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Compare the square of values for soft equality.
 */
bool SoftSurfaceEqual::soft_eq_sq(real_type a, real_type b) const
{
    return soft_eq_(std::sqrt(a), std::sqrt(b));
}

//---------------------------------------------------------------------------//
/*!
 * Compare the distance between two points for being less than the tolerance.
 *
 * The \c SoftEqual comparison is: \f[
 |a - b| < \max(\epsilon_r \max(|a|, |b|), \epsilon_a)
 \f]
 * which translates exactly to vector math.
 */
bool SoftSurfaceEqual::soft_eq_distance(Real3 const& a, Real3 const& b) const
{
    // This is soft equal formula but using vector distance.
    real_type rel = soft_eq_.abs() * std::fmax(norm(a), norm(b));
    return distance(a, b) < std::fmax(soft_eq_.abs(), rel);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
