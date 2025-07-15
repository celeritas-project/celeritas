//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/GeneratorAlgorithms.cu
//---------------------------------------------------------------------------//
#include "GeneratorAlgorithms.hh"

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/transform_scan.h>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/data/Copier.hh"
#include "corecel/math/Algorithms.hh"
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
 * Calculate the inclusive prefix sum of the number of optical photons.
 *
 * \return Total accumulated value
 */
size_type inclusive_scan_photons(
    ItemsRef<GeneratorDistributionData, MemSpace::device> const& buffer,
    ItemsRef<size_type, MemSpace::device> const& offsets,
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
}  // namespace optical
}  // namespace celeritas
