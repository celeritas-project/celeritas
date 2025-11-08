//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/ActionTimes.hh
//---------------------------------------------------------------------------//
#pragma once

#include <unordered_map>

#include "corecel/Types.hh"
#include "corecel/data/AuxInterface.hh"
#include "corecel/data/AuxStateVec.hh"

namespace celeritas
{
class ActionRegistry;
class ActionTimesState;

//---------------------------------------------------------------------------//
/*!
 * Manage state data for accumulating action times.
 */
class ActionTimes : public AuxParamsInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPActionRegistry = std::shared_ptr<ActionRegistry>;
    using MapStrDbl = std::unordered_map<std::string, double>;
    //!@}

  public:
    // Construct from aux ID and action registry
    explicit ActionTimes(AuxId, SPActionRegistry const&);

    //!@{
    //! \name Aux interface

    //! Short name for the aux data
    std::string_view label() const final { return "action-times"; }
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
    MapStrDbl action_times(AuxStateVec const&) const;

  private:
    AuxId aux_id_;
    SPActionRegistry action_reg_;
};

//---------------------------------------------------------------------------//
/*!
 * Accumulated action times on each thread.
 */
struct ActionTimesState : public AuxStateInterface
{
    using MapIdDbl = std::unordered_map<ActionId, double>;

    MapIdDbl accum_time;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
