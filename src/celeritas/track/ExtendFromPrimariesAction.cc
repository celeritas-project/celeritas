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
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/CoreTrackData.hh"

#include "generated/ProcessPrimaries.hh"

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
void ExtendFromPrimariesAction::execute_impl(CoreParams const& core_params,
                                             CoreState<M>& core_state) const
{
    auto primary_range = core_state.primary_range();
    if (primary_range.empty())
        return;

    auto primaries = core_state.primary_storage()[primary_range];

    // Create track initializers from primaries
    core_state.ref().init.scalars.num_initializers += primaries.size();
    generated::process_primaries(
        core_params.ref<M>(), core_state.ref(), primaries);

    // Mark that the primaries have been processed
    core_state.clear_primaries();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
