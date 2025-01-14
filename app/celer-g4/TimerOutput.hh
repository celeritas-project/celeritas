//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/TimerOutput.hh
//---------------------------------------------------------------------------//
#pragma once

#include <unordered_map>
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
 * Collect timing results and output at the end of a run.
 *
 * Setup time, total time, and time per event are always recorded. The
 * accumulated action times are recorded when running on the host or on the
 * device with synchronization enabled.
 */
class TimerOutput final : public OutputInterface
{
  public:
    //!@{
    //! \name Type aliases
    using MapStrReal = std::unordered_map<std::string, real_type>;
    //!@}

  public:
    // Construct with number of threads
    explicit TimerOutput(size_type num_threads);

    //!@{
    //! \name Output interface

    //! Category of data to write
    Category category() const final { return Category::result; }
    //! Key for the entry inside the category.
    std::string_view label() const final { return "time"; }
    // Write output to the given JSON object
    void output(JsonPimpl*) const final;
    //!@}

    // Record the accumulated action times
    void RecordActionTime(MapStrReal&& time);

    // Record the time for the event
    void RecordEventTime(real_type time);

    // Record the setup time
    void RecordSetupTime(real_type time);

    // Record the total time for the run
    void RecordTotalTime(real_type time);

  private:
    using VecReal = std::vector<real_type>;

    std::vector<MapStrReal> action_time_;
    std::vector<VecReal> event_time_;
    real_type setup_time_;
    real_type total_time_;
};

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
