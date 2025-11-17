//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/ActionTimes.cc
//---------------------------------------------------------------------------//
#include "ActionTimes.hh"

#include "corecel/cont/Range.hh"
#include "corecel/data/AuxParamsRegistry.hh"
#include "corecel/sys/ActionRegistry.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct and add to the aux registry.
 */
std::shared_ptr<ActionTimes>
ActionTimes::make_and_insert(SPActionRegistry const& actions,
                             SPAuxParamsRegistry const& aux,
                             std::string label)
{
    auto result = std::make_shared<ActionTimes>(
        aux->next_id(), actions, std::move(label));
    aux->insert(result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct from ID, actions and label.
 */
ActionTimes::ActionTimes(AuxId aux_id,
                         SPActionRegistry const& action_reg,
                         std::string label)
    : aux_id_(aux_id), action_reg_(action_reg), label_(std::move(label))
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
    auto result = std::make_unique<ActionTimesState>();
    auto reg = action_reg_.lock();
    result->accum_time.resize(reg->num_actions());
    return result;
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
    auto reg = action_reg_.lock();
    auto const& times = this->state(aux).accum_time;
    for (auto i : range(times.size()))
    {
        if (times[i] > 0)
        {
            result[std::string{reg->id_to_label(ActionId(i))}] = times[i];
        }
    }
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
