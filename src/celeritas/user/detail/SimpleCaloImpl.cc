//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/detail/SimpleCaloImpl.cc
//---------------------------------------------------------------------------//
#include "SimpleCaloImpl.hh"

#include "corecel/Config.hh"

#include "corecel/Types.hh"
#include "corecel/sys/KernelLauncher.hh"
#include "corecel/sys/ThreadId.hh"

#include "SimpleCaloExecutor.hh"  // IWYU pragma: associated

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Accumulate energy deposition on host.
 */
void simple_calo_accum(HostRef<StepStateData> const& step,
                       HostRef<SimpleCaloStateData>& calo)
{
    CELER_EXPECT(step && calo);
    SimpleCaloExecutor execute{step, calo};
    launch_kernel(step.size(), execute);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
