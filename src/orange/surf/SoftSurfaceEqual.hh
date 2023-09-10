//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/SoftSurfaceEqual.hh
//---------------------------------------------------------------------------//
#pragma once

#include <algorithm>

#include "corecel/cont/Range.hh"
#include "corecel/math/SoftEqual.hh"
#include "orange/OrangeTypes.hh"

#include "SurfaceFwd.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Compare two surfaces for exact equality.
 *
 * Only for exact equality should the local surface inserter return an existing
 * ID. Otherwise we could have a small gap between surfaces.
 */
struct ExactSurfaceEqual
{
    template<class S>
    inline bool operator()(S const& a, S const& b) const;
};

//---------------------------------------------------------------------------//
/*!
 * Compare two surfaces for soft equality.
 */
class SoftSurfaceEqual
{
  public:
    //! Construct with tolerances
    SoftSurfaceEqual(real_type rel, real_type abs) : soft_eq_{rel, abs} {}

    //! Construct with relative tolerance only
    explicit SoftSurfaceEqual(real_type rel) : soft_eq_{rel} {}

    //! Construct with default tolerance
    SoftSurfaceEqual() : soft_eq_{1e-10} {}

    //// SURFACE FUNCTIONS ////

    template<Axis T>
    bool operator()(PlaneAligned<T> const&, PlaneAligned<T> const&) const;

    template<Axis T>
    bool operator()(CylCentered<T> const&, CylCentered<T> const&) const;

    bool operator()(SphereCentered const&, SphereCentered const&) const;

    template<Axis T>
    bool operator()(CylAligned<T> const&, CylAligned<T> const&) const;

    bool operator()(Plane const&, Plane const&) const;

    bool operator()(Sphere const&, Sphere const&) const;

    template<Axis T>
    bool operator()(ConeAligned<T> const&, ConeAligned<T> const&) const;

    bool operator()(SimpleQuadric const&, SimpleQuadric const&) const;

    bool operator()(GeneralQuadric const&, GeneralQuadric const&) const;

  private:
    SoftEqual<real_type> soft_eq_;

    inline bool soft_eq_sq_(real_type a, real_type b) const;
    inline bool soft_eq_distance_(Real3 const& a, Real3 const& b) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Compare exact equality for two surfaces.
 */
template<class S>
bool ExactSurfaceEqual::operator()(S const& a, S const& b) const
{
    auto const& data_a = a.data();
    auto const& data_b = b.data();
    static_assert(std::is_same_v<decltype(data_a), decltype(data_b)>);
    auto r = range(data_a.size());
    return std::all_of(
        r.begin(), r.end(), [&](auto i) { return data_a[i] == data_b[i]; });
}

//---------------------------------------------------------------------------//
/*!
 * Compare the square of values for soft equality.
 *
 * \f[
 |a - b| < \max(\epsilon_r \max(|a|, |b|), \epsilon_a)
 \to
(a - b)^2 < \max(\epsilon_r^2 (\max(|a|, |b|))^2, \epsilon_a^2)
 \f]
 *
 * If \em a and \em b are \f$ O(1) \f$ and \f$ a = b \pm \epsilon \f$ then
 *
 * \f[
a^2 + b^2 - 2 a b < \epsilon'^2
a^2 + b^2 - (2 b^2 \pm 2 b \epsilon) < \epsilon'^2
a^2 - b^2 < \pm 2 b \epsilon + \epsilon'^2
|a^2 - b^2| < 2 b \epsilon + O(\epsilon'^2)
a^4 + b^4 < 2 a^2 b^2 + 4 \max(a^2, b^2) \epsilon^2
\f]
 *
 * XXX not implemented correctly
 */
bool SoftSurfaceEqual::soft_eq_sq_(real_type a, real_type b) const
{
    return SoftEqual{ipow<2>(soft_eq_.rel()), ipow<2>(soft_eq_.abs())}(a, b);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
