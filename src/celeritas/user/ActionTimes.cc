//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/ActionTimes.cc
//---------------------------------------------------------------------------//
#include "ActionTimes.hh"

#include "corecel/sys/ActionRegistry.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from ID and actions.
 */
ActionTimes::ActionTimes(AuxId aux_id, SPActionRegistry const& action_reg)
    : aux_id_(aux_id), action_reg_(action_reg)
{
    CELER_EXPECT(aux_id_);
    CELER_EXPECT(action_reg);
}

//---------------------------------------------------------------------------//
/*!
 * Build core state data for a stream.
 */
auto ActionTimes::create_state(MemSpace, StreamId, size_type) const -> UPState
{
    return std::make_unique<ActionTimesState>();
}

//---------------------------------------------------------------------------//
/*!
 * Access the state.
 */
ActionTimesState const& ActionTimes::state(AuxStateVec const& aux) const
{
    return get<ActionTimesState>(aux, aux_id_);
}

//---------------------------------------------------------------------------//
/*!
 * Access the state (mutable).
 */
ActionTimesState& ActionTimes::state(AuxStateVec& aux) const
{
    return get<ActionTimesState>(aux, aux_id_);
}

//---------------------------------------------------------------------------//
/*!
 * Create a map of action label tp accumulated time.
 */
auto ActionTimes::get_action_times(AuxStateVec const& aux) const -> MapStrDbl
{
    MapStrDbl result;
    auto sp_action_reg = action_reg_.lock();
    for (auto&& [id, time] : this->state(aux).accum_time)
    {
        result[std::string{sp_action_reg->id_to_label(id)}] = time;
    }
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
