//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/detail/OpticalGenAlgorithms.cu
//---------------------------------------------------------------------------//
#include "OpticalGenAlgorithms.hh"

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/remove.h>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "corecel/sys/Thrust.device.hh"

namespace celeritas
{
namespace optical
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Remove all invalid distributions from the buffer.
 */
size_type remove_if_invalid(Collection<GeneratorDistributionData,
                                       Ownership::reference,
                                       MemSpace::device> const& buffer,
                            size_type offset,
                            size_type size,
                            StreamId stream)
{
    ScopedProfiling profile_this{"remove-if-invalid"};
    auto start = thrust::device_pointer_cast(buffer.data().get());
    auto stop = thrust::remove_if(thrust_execute_on(stream),
                                  start + offset,
                                  start + offset + size,
                                  IsInvalid{});
    CELER_DEVICE_CHECK_ERROR();
    return stop - start;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
