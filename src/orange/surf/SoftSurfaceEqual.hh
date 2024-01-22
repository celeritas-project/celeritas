//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
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
    // Construct with tolerance
    inline SoftSurfaceEqual(Tolerance<> const& tol);

    //! Construct with relative tolerance only
    explicit SoftSurfaceEqual(real_type rel)
        : SoftSurfaceEqual{Tolerance<>::from_relative(rel)}
    {
    }

    //! Construct with default tolerance
    SoftSurfaceEqual() : SoftSurfaceEqual{Tolerance<>::from_default()} {}

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
    SoftEqual<> soft_eq_;

    bool soft_eq_sq(real_type a, real_type b) const;
    bool soft_eq_distance(Real3 const& a, Real3 const& b) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with tolerance.
 */
SoftSurfaceEqual::SoftSurfaceEqual(Tolerance<> const& tol)
    : soft_eq_{tol.rel, tol.abs}
{
    CELER_EXPECT(tol);
}

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
}  // namespace celeritas
