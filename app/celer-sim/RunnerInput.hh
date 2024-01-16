//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-sim/RunnerInput.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas_config.h"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/io/Label.hh"
#include "corecel/sys/Environment.hh"
#include "celeritas/Types.hh"
#include "celeritas/ext/GeantPhysicsOptions.hh"
#include "celeritas/ext/GeantSetup.hh"
#include "celeritas/ext/RootFileManager.hh"
#include "celeritas/field/FieldDriverOptions.hh"
#include "celeritas/phys/PrimaryGeneratorOptions.hh"
#include "celeritas/user/RootStepWriter.hh"

#ifdef _WIN32
#    include <cstdlib>
#    ifdef environ
#        undef environ
#    endif
#endif

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Input for a single run.
 */
struct RunnerInput
{
    static constexpr Real3 no_field() { return Real3{0, 0, 0}; }
    static constexpr size_type unspecified{static_cast<size_type>(-1)};

    // Global environment
    size_type cuda_heap_size{unspecified};
    size_type cuda_stack_size{unspecified};
    Environment environ;  //!< Supplement existing env variables

    // Problem definition
    std::string geometry_file;  //!< Path to GDML file
    std::string physics_file;  //!< Path to ROOT exported Geant4 data
    std::string event_file;  //!< Path to input event data

    // Optional setup options for generating primaries programmatically
    PrimaryGeneratorOptions primary_options;

    // Diagnostics and output
    std::string mctruth_file;  //!< Path to ROOT MC truth event data
    SimpleRootFilterInput mctruth_filter;
    std::vector<Label> simple_calo;
    bool action_diagnostic{};
    bool step_diagnostic{};
    int step_diagnostic_bins{1000};
    bool write_track_counts{true};  //!< Output track counts for each step
    bool write_step_times{true};  //!< Output elapsed times for each step

    // Control
    unsigned int seed{};
    size_type num_track_slots{};  //!< Divided among streams
    size_type max_steps{unspecified};
    size_type initializer_capacity{};  //!< Divided among streams
    real_type secondary_stack_factor{};
    bool use_device{};
    bool sync{};
    bool merge_events{false};  //!< Run all events at once on a single stream
    bool default_stream{false};  //!< Launch all kernels on the default stream
    bool warm_up{CELER_USE_DEVICE};  //!< Run a nullop step first

    // Magnetic field vector [* 1/Tesla] and associated field options
    Real3 field{no_field()};
    FieldDriverOptions field_options;

    // Optional fixed-size step limiter for charged particles
    // (non-positive for unused)
    real_type step_limiter{};

    // Options for physics
    bool brem_combined{false};

    // Track init options
    TrackOrder track_order{TrackOrder::unsorted};

    // Optional setup options if loading directly from Geant4
    GeantPhysicsOptions physics_options;

    //! Whether the run arguments are valid
    explicit operator bool() const
    {
        return !geometry_file.empty()
               && (primary_options || !event_file.empty())
               && num_track_slots > 0 && max_steps > 0
               && initializer_capacity > 0 && secondary_stack_factor > 0
               && (step_diagnostic_bins > 0 || !step_diagnostic)
               && (field == no_field() || field_options);
    }
};

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
