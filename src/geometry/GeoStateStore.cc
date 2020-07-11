//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeoStateStore.cu
//---------------------------------------------------------------------------//
#include "GeoStateStore.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with device geometry and number of state elements.
 */
GeoStateStore::GeoStateStore(SptrConstParams geom, size_type size)
    : geom_(std::move(geom))
{
    REQUIRE(geom_);
    REQUIRE(size > 0);

    vgstate_   = detail::VGNavStateStore(size, geom_->max_depth());
    vgnext_    = detail::VGNavStateStore(size, geom_->max_depth());
    pos_       = DeviceVector<Real3>(size);
    dir_       = DeviceVector<Real3>(size);
    next_step_ = DeviceVector<double>(size);
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get a view to on-device states
 */
GeoStatePointers GeoStateStore::device_pointers()
{
    GeoStatePointers result;
    result.size       = this->size();
    result.vgmaxdepth = geom_->max_depth();
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
