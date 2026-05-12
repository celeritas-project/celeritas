//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/StepTimes.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <vector>

#include "corecel/Types.hh"
#include "corecel/data/AuxInterface.hh"
#include "corecel/data/AuxStateVec.hh"

namespace celeritas
{
class AuxParamsRegistry;
struct StepTimesState;

//---------------------------------------------------------------------------//
/*!
 * Manage state data for recording step times.
 *
 * This class allocates thread-local state data that can be used to record the
 * time of each step on each stream over the run. Because the class that
 * invokes the sequence of steps should be shared across threads, the step
 * times are stored as auxiliary data rather than locally in that class.
 */
class StepTimes : public AuxParamsInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPAuxParamsRegistry = std::shared_ptr<AuxParamsRegistry>;
    using VecDbl = std::vector<double>;
    //!@}

  public:
    // Construct and add to the aux registry
    static std::shared_ptr<StepTimes>
    make_and_insert(SPAuxParamsRegistry const&, std::string label);

    // Construct from ID and label
    StepTimes(AuxId, std::string label);

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
    StepTimesState const& state(AuxStateVec const&) const;
    StepTimesState& state(AuxStateVec&) const;

  private:
    AuxId aux_id_;
    std::string label_;
};

//---------------------------------------------------------------------------//
/*!
 * Step times on each thread.
 */
struct StepTimesState : public AuxStateInterface
{
    StepTimes::VecDbl time;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
