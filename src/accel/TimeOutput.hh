//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/TimeOutput.hh
//---------------------------------------------------------------------------//
#pragma once

#include <unordered_map>
#include <vector>

#include "corecel/Types.hh"
#include "corecel/io/OutputInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Collect timing results and output at the end of a run.
 *
 * Setup time and total time are always recorded. Event time is recorded if \c
 * BeginOfEventAction and \c EndOfEventAction are called. The accumulated
 * action times are recorded when running on the host or on the device with
 * synchronization enabled.
 *
 * All results are in units of seconds.
 */
class TimeOutput final : public OutputInterface
{
  public:
    //!@{
    //! \name Type aliases
    using MapStrReal = std::unordered_map<std::string, real_type>;
    //!@}

  public:
    // Construct with number of CPU threads
    explicit TimeOutput(size_type num_threads);

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

    // Record the Celeritas setup time
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
}  // namespace celeritas
