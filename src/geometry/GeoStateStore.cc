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
    vgstate_   = detail::VGNavStateStore(size, max_depth_);
    vgnext_    = detail::VGNavStateStore(size, max_depth_);
    pos_       = DeviceVector<Real3>(size);
    dir_       = DeviceVector<Real3>(size);
    next_step_ = DeviceVector<double>(size);
}

//---------------------------------------------------------------------------//
/*!
 * Initialize states on the host.
 *
 * This must currently be done independently of the device.
 */
void GeoStateStore::initialize(const GeoParams&,
                               SpanConstReal3 pos,
                               SpanConstReal3 dir)
{
    REQUIRE(pos.size() == this->size());
    REQUIRE(dir.size() == this->size());

    auto& cur_state_pool  = vgstate_.get();
    auto& next_state_pool = vgnext_.get();

    // Initialize "current" state using host pointers
    const vecgeom::VPlacedVolume* host_world
        = vecgeom::GeoManager::Instance().GetWorld();
    constexpr bool contains_point = true;
    for (auto i : range(this->size()))
    {
        cur_state_pool[i]->Clear();
        vecgeom::GlobalLocator::LocateGlobalPoint(host_world,
                                                  detail::to_vector(pos[i]),
                                                  *cur_state_pool[i],
                                                  contains_point);
        next_state_pool[i]->Clear();
    }

    // Clear 'next step'
    std::vector<real_type> next_step(
        this->size(), celeritas::numeric_limits<real_type>::quiet_NaN());

    // Copy data to device
    vgstate_.copy_to_device();
    vgnext_.copy_to_device();
    pos_.copy_to_device(pos);
    dir_.copy_to_device(dir);
    next_step_.copy_to_device(make_span(next_step));
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get a view to on-device states
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

    ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
