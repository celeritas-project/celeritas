//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/SoftSurfaceEqual.hh
//! \todo Move to orange construction directory
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
 * Compare two surfaces for soft equality.
 *
 * Ideally, this would evaluate whether the Hausdorff distance between two
 * surfaces, within some bounding box, is less than the tolerance.
 */
class SoftSurfaceEqual
{
  public:
    // Construct with tolerance
    explicit inline SoftSurfaceEqual(Tolerance<> const& tol);

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

    bool operator()(Involute const&, Involute const&) const;

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
}  // namespace celeritas
