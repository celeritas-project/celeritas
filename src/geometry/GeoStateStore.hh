//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeoStateStore.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include "base/Array.hh"
#include "base/DeviceVector.hh"
#include "base/Span.hh"
#include "base/Types.hh"
#include "GeoStatePointers.hh"
#include "detail/VGNavStateStore.hh"

namespace celeritas
{
class GeoParams;
//---------------------------------------------------------------------------//
/*!
 * Manage on-device VecGeom states.
 *
 * \note Construction on a build without a GPU (or CUDA) will raise an error.
 */
class GeoStateStore
{
  public:
    //@{
    //! Public types.
    using SpanConstReal3 = span<const Real3>;
    //@}

  public:
    // Construct from device geometry and number of track states
    GeoStateStore(const GeoParams& geo, size_type size);

    // Locate positions/directions on host, and copy to device
    // XXX temporary; delete once GlobalLocator works
    void initialize(const GeoParams&, SpanConstReal3 pos, SpanConstReal3 dir);

    // >>> ACCESSORS

    // Number of states
    size_type size() const { return pos_.size(); }

    // View on-device states
    GeoStatePointers device_pointers();

  private:
    int                     max_depth_;
    detail::VGNavStateStore vgstate_;
    detail::VGNavStateStore vgnext_;
    DeviceVector<Real3>     pos_;
    DeviceVector<Real3>     dir_;
    DeviceVector<double>    next_step_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
