//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ValueGridStore.cc
//---------------------------------------------------------------------------//
#include "ValueGridStore.hh"

#include "base/SpanRemapper.hh"
#include "base/VectorUtils.hh"
#include "comm/Device.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with storage space requirements.
 *
 * This allocates device data and resizes it to empty (so that the vector can
 * keep track of its actual size).
 */
ValueGridStore::ValueGridStore(size_type num_grids, size_type num_values)
{
    CELER_EXPECT(num_grids > 0);
    CELER_EXPECT(num_values > 0);
    host_grids_.reserve(num_grids);
    host_values_.reserve(num_values);
}

//---------------------------------------------------------------------------//
/*!
 * Add a grid of host pointer data.
 */
void ValueGridStore::push_back(const XsGridPointers& inp)
{
    CELER_EXPECT(inp);
    CELER_EXPECT(this->size() != this->capacity());
    CELER_EXPECT(host_values_.size() + inp.value.size()
                 <= host_values_.capacity());

    host_grids_.push_back(inp);
    host_grids_.back().value = celeritas::extend(inp.value, &host_values_);
}

//---------------------------------------------------------------------------//
/*!
 * Copy all data to device.
 */
void ValueGridStore::copy_to_device()
{
    CELER_EXPECT(!host_grids_.empty());
    CELER_EXPECT(celeritas::is_device_enabled());

    device_grids_  = DeviceVector<XsGridPointers>(host_grids_.size());
    device_values_ = DeviceVector<real_type>(host_values_.size());

    device_grids_.copy_to_device(make_span(host_grids_));
    device_values_.copy_to_device(make_span(host_values_));
}

//---------------------------------------------------------------------------//
/*!
 * Get host data.
 */
auto ValueGridStore::host_pointers()  const -> ValueGridPointers
{
    return make_span(host_grids_);
}

//---------------------------------------------------------------------------//
/*!
 * Get device data.
 */
auto ValueGridStore::device_pointers() const -> ValueGridPointers
{
    CELER_EXPECT(!device_grids_.empty());
    return device_grids_.device_pointers();
}

//---------------------------------------------------------------------------//
} // namespace celeritas
