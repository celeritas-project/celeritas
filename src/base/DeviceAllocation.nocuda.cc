//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file DeviceAllocation.nocuda.cc
//---------------------------------------------------------------------------//
#include "DeviceAllocation.hh"
#include "Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Prevent allocation because CUDA is disabled.
 */
DeviceAllocation::DeviceAllocation(size_type)
{
    CELER_NOT_CONFIGURED("CUDA");
}

//---------------------------------------------------------------------------//
//! Deleter should never be called on CPU
void DeviceAllocation::CudaFreeDeleter::operator()(Byte*) const
{
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
/*!
 * Copy data to device (not implemented when cuda is disabled).
 */
void DeviceAllocation::copy_to_device(SpanConstBytes)
{
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
/*!
 * Copy data to host (not implemented when cuda is disabled).
 */
void DeviceAllocation::copy_to_host(SpanBytes) const
{
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
} // namespace celeritas
