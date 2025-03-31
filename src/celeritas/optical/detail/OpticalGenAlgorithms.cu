//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/OpticalGenAlgorithms.cu
//---------------------------------------------------------------------------//
#include "OpticalGenAlgorithms.hh"

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/remove.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform_scan.h>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/data/Copier.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "corecel/sys/Thrust.device.hh"

namespace celeritas
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//

struct GetNumPhotons
{
    // Return the number of photons to generate
    CELER_FUNCTION size_type
    operator()(celeritas::optical::GeneratorDistributionData const& data) const
    {
        return data.num_photons;
    }
};

//---------------------------------------------------------------------------//
}  // namespace

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
        thrust_execute_on(stream), start + offset, start + size, IsInvalid{});
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
    size_type count = thrust::transform_reduce(thrust_execute_on(stream),
                                               start + offset,
                                               start + size,
                                               GetNumPhotons{},
                                               size_type(0),
                                               thrust::plus<size_type>());
    CELER_DEVICE_API_CALL(PeekAtLastError());
    return count;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the inclusive prefix sum of the number of optical photons.
 *
 * \return Total accumulated value
 */
size_type inclusive_scan_photons(
    GeneratorDistributionRef<MemSpace::device> const& buffer,
    Collection<size_type, Ownership::reference, MemSpace::device> const& offsets,
    size_type size,
    StreamId stream)
{
    CELER_EXPECT(!buffer.empty());
    CELER_EXPECT(size > 0 && size <= buffer.size());
    CELER_EXPECT(offsets.size() == buffer.size());

    ScopedProfiling profile_this{"inclusive-scan-photons"};
    auto data = thrust::device_pointer_cast(buffer.data().get());
    auto result = thrust::device_pointer_cast(offsets.data().get());
    auto stop = thrust::transform_inclusive_scan(thrust_execute_on(stream),
                                                 data,
                                                 data + size,
                                                 result,
                                                 GetNumPhotons{},
                                                 thrust::plus<size_type>());
    CELER_DEVICE_API_CALL(PeekAtLastError());

    // Copy the last element (accumulated total) back to host
    return ItemCopier<size_type>{stream}(stop.get() - 1);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
