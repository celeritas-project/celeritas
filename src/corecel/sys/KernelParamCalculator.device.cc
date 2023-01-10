//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/KernelParamCalculator.device.cc
//---------------------------------------------------------------------------//
#include "KernelParamCalculator.device.hh"

#include "KernelRegistry.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Add a new kernel to the registry.
 */
void KernelParamCalculator::register_kernel(char const* name,
                                            KernelAttributes&& attributes)
{
    profiling_
        = celeritas::kernel_registry().insert(name, std::move(attributes));
}

//---------------------------------------------------------------------------//
/*!
 * Accumulate counters for a kernel launch.
 */
void KernelParamCalculator::log_launch(size_type min_num_threads) const
{
    CELER_EXPECT(profiling_);
    profiling_->log_launch(min_num_threads);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
