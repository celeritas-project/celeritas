//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/TimerOutput.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>
#include <G4Event.hh>

#include "corecel/Types.hh"
#include "corecel/io/OutputInterface.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Record the total time, accumulated time per action, and time per event.
 */
class TimerOutput final : public OutputInterface
{
  public:
    //!@{
    //! \name Type aliases
    using VecReal = std::vector<real_type>;
    //!@}

  public:
    // Construct with number of events
    explicit TimerOutput(size_type num_events);

    //!@{
    //! \name Output interface
    //! Category of data to write
    Category category() const final { return Category::result; }
    //! Key for the entry inside the category.
    std::string label() const final { return "time"; }
    // Write output to the given JSON object
    void output(JsonPimpl*) const final;
    //!@}

    // Record the total time for the run
    void RecordTotalTime(real_type time);

    // Record the time for the given event
    void RecordEventTime(G4Event const* event, real_type time);

  private:
    real_type total_time_;
    VecReal event_time_;
};

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
