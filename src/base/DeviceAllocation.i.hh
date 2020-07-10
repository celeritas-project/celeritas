//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file DeviceAllocation.i.hh
//---------------------------------------------------------------------------//

#include "Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Swap with another allocation.
 */
void DeviceAllocation::swap(DeviceAllocation& other) noexcept
{
    using std::swap;
    swap(this->data_, other.data_);
    swap(this->size_, other.size_);
}

//---------------------------------------------------------------------------//
/*!
 * Get a view to the owned device memory.
 */
auto DeviceAllocation::device_view() -> SpanBytes
{
    return {data_.get(), size_};
}

//---------------------------------------------------------------------------//
/*!
 * Get a view to the owned device memory.
 */
auto DeviceAllocation::device_view() const -> constSpanBytes
{
    return {data_.get(), size_};
}

//---------------------------------------------------------------------------//
/*!
 * Swap two allocations.
 */
void swap(DeviceAllocation& a, DeviceAllocation& b) noexcept
{
    return a.swap(b);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
