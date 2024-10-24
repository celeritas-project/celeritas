//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-sim/RunnerInput.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Config.hh"

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
 *
 * TODO for v1.0: unify these names, combine with celer-g4, separate into
 * schemas for individual classes, ... ? and decide whether max_steps should be
 * per track or total step iterations.
 */
struct RunnerInput
{
    struct EventFileSampling
    {
        size_type num_events{};  //!< Total number of events to sample
        size_type num_merged{};  //!< ROOT file events per sampled event

        explicit operator bool() const
        {
            return num_events > 0 && num_merged > 0;
        };
    };

    struct OpticalOptions
    {
        size_type buffer_capacity{};  //!< Number of steps that created photons
        size_type primary_capacity{};  //!< Maximum number of pending primaries
        size_type auto_flush{};  //!< Threshold number of primaries for
                                 //!< launching optical tracking loop

        explicit operator bool() const
        {
            return buffer_capacity > 0 && primary_capacity > 0
                   && auto_flush > 0;
        };
    };
    static constexpr Real3 no_field() { return Real3{0, 0, 0}; }
    static constexpr size_type unspecified{0};

    // Global environment
    size_type cuda_heap_size{unspecified};
    size_type cuda_stack_size{unspecified};
    Environment environ;  //!< Supplement existing env variables

    // Problem definition
    std::string geometry_file;  //!< Path to GDML file
    std::string physics_file;  //!< Path to ROOT exported Geant4 data
    std::string event_file;  //!< Path to input event data

    // Optional setup when event_file is a ROOT input used for sampling
    // combinations of events as opposed to just reading them
    EventFileSampling file_sampling_options;  //!< ROOT sampling options

    // Optional setup options for generating primaries programmatically
    PrimaryGeneratorOptions primary_options;

    // Diagnostics and output
    std::string mctruth_file;  //!< Path to ROOT MC truth event data
    std::string tracing_file;
    SimpleRootFilterInput mctruth_filter;
    std::vector<Label> simple_calo;
    bool action_diagnostic{};
    bool step_diagnostic{};
    int step_diagnostic_bins{1000};
    std::string slot_diagnostic_prefix;  //!< Base name for slot diagnostic
    bool write_track_counts{true};  //!< Output track counts for each step
    bool write_step_times{true};  //!< Output elapsed times for each step

    // Control
    unsigned int seed{};
    size_type num_track_slots{};  //!< Divided among streams
    size_type max_steps = static_cast<size_type>(-1);
    size_type initializer_capacity{};  //!< Divided among streams
    real_type secondary_stack_factor{};
    bool use_device{};
    bool action_times{};
    bool merge_events{false};  //!< Run all events at once on a single stream
    bool default_stream{false};  //!< Launch all kernels on the default stream
    bool warm_up{false};  //!< Run a nullop step first

    // Magnetic field vector [* 1/Tesla] and associated field options
    Real3 field{no_field()};
    FieldDriverOptions field_options;

    // Optional fixed-size step limiter for charged particles
    // (non-positive for unused)
    real_type step_limiter{};

    // Options for physics
    bool brem_combined{false};

    // Track reordering options
    TrackOrder track_order{TrackOrder::none};

    // Optional setup options if loading directly from Geant4
    GeantPhysicsOptions physics_options;

    // Options when optical physics is enabled
    OpticalOptions optical;

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
