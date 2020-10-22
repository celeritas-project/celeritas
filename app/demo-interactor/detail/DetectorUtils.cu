//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file DetectorUtils.cu
//---------------------------------------------------------------------------//
#include "DetectorUtils.hh"

#include "base/Atomics.hh"
#include "base/KernelParamCalculator.cuda.hh"
#include "base/StackAllocatorView.hh"

namespace
{
//---------------------------------------------------------------------------//
using namespace celeritas;

__global__ void bin_buffer_kernel(DetectorPointers const detector)
{
    auto        hits = StackAllocatorView<Hit>(detector.hit_buffer).get();
    UniformGrid grid(detector.tally_grid);
    size_type   thread_idx = KernelParamCalculator::thread_id().get();

    if (thread_idx < hits.size())
    {
        // Find bin
        const Hit& hit   = hits[thread_idx];
        real_type  z_pos = hit.pos[2];
        size_type  bin;
        if (z_pos <= grid.front())
            bin = 0;
        else if (z_pos >= grid.back())
            bin = grid.size() - 1;
        else
            bin = grid.find(z_pos);

        // Add energy deposition (NOTE: very slow on arch 600)
        atomic_add(&detector.tally_deposition[bin],
                   hit.energy_deposited.value());
    }
}

//---------------------------------------------------------------------------//

__global__ void
normalize_kernel(DetectorPointers const detector, real_type norm)
{
    size_type thread_idx = KernelParamCalculator::thread_id().get();
    if (thread_idx < detector.tally_deposition.size())
    {
        detector.tally_deposition[thread_idx] *= norm;
    }
}

//---------------------------------------------------------------------------//
} // namespace

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Bin the buffer into the tally grid.
 *
 * The caller will have to clear the buffer after calling this. No
 * normalization is performed.
 */
void bin_buffer(const DetectorPointers& detector)
{
    auto params = KernelParamCalculator()(detector.capacity());
    bin_buffer_kernel<<<params.grid_size, params.block_size>>>(detector);
}

//---------------------------------------------------------------------------//
/*!
 * Multiply the binned data by the given normalization.
 */
void normalize(const DetectorPointers& device_ptrs, real_type norm)
{
    auto params = KernelParamCalculator()(device_ptrs.tally_deposition.size());
    normalize_kernel<<<params.grid_size, params.block_size>>>(device_ptrs,
                                                              norm);
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
