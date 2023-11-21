//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/ExtendFromPrimariesAction.cc
//---------------------------------------------------------------------------//
#include "ExtendFromPrimariesAction.hh"

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"

#include "detail/ProcessPrimariesExecutor.hh"  // IWYU pragma: associated

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Execute the action with host data.
 */
void ExtendFromPrimariesAction::execute(CoreParams const& params,
                                        CoreStateHost& state) const
{
    return this->execute_impl(params, state);
}

//---------------------------------------------------------------------------//
/*!
 * Execute the action with device data.
 */
void ExtendFromPrimariesAction::execute(CoreParams const& params,
                                        CoreStateDevice& state) const
{
    return this->execute_impl(params, state);
}

//---------------------------------------------------------------------------//
/*!
 * Construct primaries.
 */
template<MemSpace M>
void ExtendFromPrimariesAction::execute_impl(CoreParams const& params,
                                             CoreState<M>& state) const
{
    auto primary_range = state.primary_range();
    if (primary_range.empty() && !state.warming_up())
        return;

    auto primaries = state.primary_storage()[primary_range];

    // Create track initializers from primaries
    state.counters().num_initializers += primaries.size();
    this->process_primaries(params, state, primaries);

    // Mark that the primaries have been processed
    state.clear_primaries();
}

//---------------------------------------------------------------------------//
/*!
 * Launch a (host) kernel to create track initializers from primary particles.
 */
void ExtendFromPrimariesAction::process_primaries(
    CoreParams const& params,
    CoreStateHost& state,
    Span<Primary const> primaries) const
{
    launch_action(*this,
                  primaries.size(),
                  params,
                  state,
                  detail::ProcessPrimariesExecutor{
                      state.ptr(), primaries, state.counters()});
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void ExtendFromPrimariesAction::process_primaries(CoreParams const&,
                                                  CoreStateDevice&,
                                                  Span<Primary const>) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
