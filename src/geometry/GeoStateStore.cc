//---------------------------------*-C++-*-----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeoStateStore.cc
//---------------------------------------------------------------------------//
#include "GeoStateStore.hh"

#include <VecGeom/management/GeoManager.h>
#include <VecGeom/navigation/GlobalLocator.h>
#include <VecGeom/navigation/NavStatePool.h>
#include "base/Range.hh"
#include "base/NumericLimits.hh"
#include "base/Types.hh"
#include "comm/Device.hh"
#include "GeoParams.hh"
#include "detail/VGCompatibility.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with device geometry and number of state elements.
 */
GeoStateStore::GeoStateStore(const GeoParams& geom, size_type size)
    : max_depth_(geom.max_depth())
{
    CELER_EXPECT(celeritas::is_device_enabled());
    vgstate_   = detail::VGNavStateStore(size, max_depth_);
    vgnext_    = detail::VGNavStateStore(size, max_depth_);
    pos_       = DeviceVector<Real3>(size);
    dir_       = DeviceVector<Real3>(size);
    next_step_ = DeviceVector<double>(size);
}

//---------------------------------------------------------------------------//
/*!
 * Get a view to on-device states.
 */
GeoStatePointers GeoStateStore::device_pointers()
{
    GeoStatePointers result;
    result.size       = this->size();
    result.vgmaxdepth = max_depth_;
    result.vgstate    = vgstate_.device_pointers();
    result.vgnext     = vgnext_.device_pointers();
    result.pos        = pos_.device_pointers().data();
    result.dir        = dir_.device_pointers().data();
    result.next_step  = next_step_.device_pointers().data();

    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
