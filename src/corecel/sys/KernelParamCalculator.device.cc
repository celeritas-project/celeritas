//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/KernelParamCalculator.device.cc
//---------------------------------------------------------------------------//
#include "KernelParamCalculator.device.hh"

#include <cassert>
#include <limits>

#include "corecel/device_runtime_api.h"
#include "corecel/Assert.hh"

namespace
{
//---------------------------------------------------------------------------//
// Integer division, rounding up, for positive numbers
template<class UInt>
UInt ceil_div(UInt top, UInt bottom)
{
    return (top / bottom) + (top % bottom != 0);
}
//---------------------------------------------------------------------------//
} // namespace

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Calculate launch params given the number of threads.
 */
KernelParamCalculator::LaunchParams
KernelParamCalculator::operator()(size_type min_num_threads) const
{
    CELER_EXPECT(min_num_threads > 0);
    CELER_EXPECT(min_num_threads <= static_cast<size_type>(
                     std::numeric_limits<dim_type>::max()));

    // Update diagnostics for the kernel
    celeritas::kernel_diagnostics().launch(id_, min_num_threads);

    // Ceiling integer division
    dim_type blocks_per_grid
        = ceil_div<dim_type>(min_num_threads, this->block_size_);

    LaunchParams result;
    result.blocks_per_grid.x   = blocks_per_grid;
    result.threads_per_block.x = this->block_size_;
    CELER_ENSURE(result.blocks_per_grid.x * result.threads_per_block.x
                 >= min_num_threads);
    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
