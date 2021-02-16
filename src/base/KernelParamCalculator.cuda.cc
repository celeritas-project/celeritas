//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file KernelParamCalculator.cuda.cc
//---------------------------------------------------------------------------//
#include <cassert>
#include <limits>

#include "KernelParamCalculator.cuda.hh"
#include "Assert.hh"

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
 *
 * \todo Once we start worrying about multiple devices on a single machine, we
 * could add a check for the device ID being the same one used to initialize
 * the param calculator.
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
    dim_type grid_size = ceil_div<dim_type>(min_num_threads, this->block_size_);

    LaunchParams result;
    result.grid_size.x  = grid_size;
    result.block_size.x = this->block_size_;
    CELER_ENSURE(result.grid_size.x * result.block_size.x >= min_num_threads);
    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
