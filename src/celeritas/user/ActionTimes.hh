//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/ActionTimes.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "corecel/Types.hh"
#include "corecel/data/AuxInterface.hh"
#include "corecel/data/AuxStateVec.hh"

namespace celeritas
{
class ActionRegistry;
struct ActionTimesState;
class AuxParamsRegistry;

//---------------------------------------------------------------------------//
/*!
 * Manage state data for accumulating action times.
 *
 * This class allocates thread-local state data that can be used to accumulate
 * the time of each step action on each stream over the run. Because the class
 * that invokes the sequence of step actions should be shared across threads,
 * the action times are stored as auxiliary data rather than locally in that
 * class.
 */
class ActionTimes : public AuxParamsInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPActionRegistry = std::shared_ptr<ActionRegistry>;
    using SPAuxParamsRegistry = std::shared_ptr<AuxParamsRegistry>;
    using MapStrDbl = std::unordered_map<std::string, double>;
    //!@}

  public:
    // Construct and add to the aux registry
    static std::shared_ptr<ActionTimes>
    make_and_insert(SPActionRegistry const&,
                    SPAuxParamsRegistry const&,
                    std::string label);

    // Construct from ID, actions and label
    ActionTimes(AuxId, SPActionRegistry const&, std::string label);

    //!@{
    //! \name Aux interface

    //! Short name for the aux data
    std::string_view label() const final { return label_; }
    //! Index of this class instance in its registry
    AuxId aux_id() const final { return aux_id_; }
    // Build core state data for a stream
    UPState create_state(MemSpace, StreamId, size_type) const final;
    //!@}

    // Access the state
    ActionTimesState const& state(AuxStateVec const&) const;

    // Access the state (mutable)
    ActionTimesState& state(AuxStateVec&) const;

    // Create a map of action label tp accumulated time
    MapStrDbl get_action_times(AuxStateVec const&) const;

  private:
    AuxId aux_id_;
    std::weak_ptr<ActionRegistry> action_reg_;
    std::string label_;
};

//---------------------------------------------------------------------------//
/*!
 * Accumulated action times on each thread.
 *
 * \todo Always report CPU times and add a second vector for device runs that
 * uses the CUDA event API to record GPU times.
 */
struct ActionTimesState : public AuxStateInterface
{
    std::vector<double> accum_time;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
