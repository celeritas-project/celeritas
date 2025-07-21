//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/OffloadAlgorithms.cu
//---------------------------------------------------------------------------//
#include "OffloadAlgorithms.hh"

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/remove.h>
#include <thrust/transform_reduce.h>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/data/Copier.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "corecel/sys/Thrust.device.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Remove all invalid distributions from the buffer.
 *
 * \return Total number of valid distributions in the buffer
 */
size_type
remove_if_invalid(GeneratorDistributionRef<MemSpace::device> const& buffer,
                  size_type offset,
                  size_type size,
                  StreamId stream)
{
    ScopedProfiling profile_this{"remove-if-invalid"};
    auto start = thrust::device_pointer_cast(buffer.data().get());
    auto stop = thrust::remove_if(
        thrust_execute_on(stream), start + offset, start + size, LogicalFalse{});
    CELER_DEVICE_API_CALL(PeekAtLastError());
    return stop - start;
}

//---------------------------------------------------------------------------//
/*!
 * Count the number of optical photons in the distributions.
 */
size_type
count_num_photons(GeneratorDistributionRef<MemSpace::device> const& buffer,
                  size_type offset,
                  size_type size,
                  StreamId stream)
{
    ScopedProfiling profile_this{"count-num-photons"};
    auto start = thrust::device_pointer_cast(buffer.data().get());
    size_type count
        = thrust::transform_reduce(thrust_execute_on(stream),
                                   start + offset,
                                   start + size,
                                   celeritas::optical::GetNumPhotons{},
                                   size_type(0),
                                   thrust::plus<size_type>());
    CELER_DEVICE_API_CALL(PeekAtLastError());
    return count;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
