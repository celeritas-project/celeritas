//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/Diagnostics.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>
#include <optional>
#include <string>

#include "corecel/Types.hh"
#include "celeritas/user/RootStepWriterInput.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class CoreParams;

namespace inp
{
//---------------------------------------------------------------------------//
/*!
 * Write out problem data to separate files for debugging.
 *
 * These options are meant for use in the context of a larger experiment
 * framework, for exporting physics settings, detector geometry, and offloaded
 * EM tracks for reproducing externally.
 *
 * \todo Some of these will be one per run; others will be one per thread.
 */
struct ExportFiles
{
    //! Filename for ROOT dump of physics data
    std::string physics;
    //! Filename to dump a ROOT/HepMC3 copy of primaries
    std::string offload;
    //! Filename to dump a GDML file of the active Geant4 geometry
    std::string geometry;
};

//---------------------------------------------------------------------------//
/*!
 * Export (possibly large!) diagnostic output about track slot contents.
 *
 * \sa celeritas::SlotDiagnostic
 */
struct SlotDiagnostic
{
    //! Prefix of file names for outputting on each stream
    std::string basename;
};

//---------------------------------------------------------------------------//
/*!
 * Set up Celeritas timers.
 */
struct Timers
{
    //! Accumulate elapsed time for each action
    bool action{false};
    //! Save elapsed time for each step
    bool step{false};
};

//---------------------------------------------------------------------------//
/*!
 * Output track diagnostic counters.
 *
 * These include the number of tracks generated, active, aborted, and alive;
 * as well as the number of initializers (or the high water mark thereof).
 */
struct Counters
{
    //! Write diagnostics for each step
    bool step{false};
    //! Write diagnostics for each event (or run, if multiple events)
    bool event{false};
};

//---------------------------------------------------------------------------//
/*!
 * Write out MC truth data.
 *
 * \sa celeritas::RootStepWriter
 */
struct McTruth
{
    //! Path to saved ROOT mc truth file
    std::string output_file;

    //! Filter saved data by track ID, particle type
    SimpleRootFilterInput filter;
};

//---------------------------------------------------------------------------//
/*!
 * Accumulate distributions of the number of steps per particle type.
 */
struct StepDiagnostic
{
    //! Maximum number of steps per track to bin
    size_type bins{1000};
};

//---------------------------------------------------------------------------//
/*!
 * Set up Celeritas built-in diagnostics.
 */
struct Diagnostics
{
    //! Write Celeritas diagnostics to this file ("-" is stdout)
    std::string output_file{"-"};

    //! Export problem setup
    ExportFiles export_files;

    //! Write elapsed times for each step
    Timers timers;

    //! Store step/track counts
    Counters counters;

    //! Write Perfetto tracing data to this filename
    std::string perfetto_file;

    //! Activate slot diagnostics
    std::optional<SlotDiagnostic> slot;

    /*!
     * Accumulate post-step actions for each particle type.
     *
     * \sa celeritas::ActionDiagnostic
     */
    bool action{false};

    //! Add a 'status checker' for debugging new actions
    bool status_checker{false};

    //! Write detailed MC truth output
    std::optional<McTruth> mctruth;

    //! Bin number of steps per track
    std::optional<StepDiagnostic> step;

    //! Log the execution progress every N events
    size_type log_frequency{1};

    //! Add additional diagnostic user actions
    std::function<void(CoreParams const&)> add_user_actions;
};

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
