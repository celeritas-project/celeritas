//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeoStateStore.hh
//---------------------------------------------------------------------------//
#ifndef geometry_VGStateStore_hh
#define geometry_VGStateStore_hh

#include <memory>
#include "base/Array.hh"
#include "base/DeviceVector.hh"
#include "base/Types.hh"
#include "GeoParams.hh"
#include "GeoStatePointers.hh"
#include "detail/VGNavStateStore.hh"

namespace celeritas
{
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
    //! Type aliases
    using SptrConstParams = std::shared_ptr<const GeoParams>;
    //@}

  public:
    // Construct from device geometry and number of track states
    GeoStateStore(SptrConstParams geom, size_type size);

    // >>> ACCESSORS

    // Number of states
    size_type size() const { return pos_.size(); }

    // View on-device states
    GeoStatePointers device_pointers();

  private:
    SptrConstParams geom_;

    detail::VGNavStateStore vgstate_;
    detail::VGNavStateStore vgnext_;
    DeviceVector<Real3>     pos_;
    DeviceVector<Real3>     dir_;
    DeviceVector<double>    next_step_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#endif // geometry_VGStateStore_hh
