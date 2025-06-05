//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-sim/RunnerInput.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <vector>

#include "corecel/Config.hh"

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/io/Label.hh"
#include "corecel/sys/Environment.hh"
#include "celeritas/Types.hh"
#include "celeritas/ext/GeantPhysicsOptions.hh"
#include "celeritas/ext/RootFileManager.hh"
#include "celeritas/field/FieldDriverOptions.hh"
#include "celeritas/phys/PrimaryGeneratorOptions.hh"
#include "celeritas/user/RootStepWriter.hh"
#include "celeritas/user/RootStepWriterInput.hh"

#ifdef _WIN32
#    include <cstdlib>
#    ifdef environ
#        undef environ
#    endif
#endif

namespace celeritas
{
namespace inp
{
struct StandaloneInput;
}

namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Input for a single run.
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
        }
    };

    struct OpticalOptions
    {
        // Sizes are divided among streams
        size_type num_track_slots{};  //!< Number of optical loop tracks slots
        size_type buffer_capacity{};  //!< Number of steps that created photons
        size_type initializer_capacity{};  //!< Maximum queued tracks
        size_type auto_flush{};  //!< Threshold number of primaries for
                                 //!< launching optical tracking loop
        size_type max_steps = static_cast<size_type>(-1);  //!< Step iterations

        explicit operator bool() const
        {
            return num_track_slots > 0 && buffer_capacity > 0
                   && initializer_capacity > 0 && auto_flush > 0
                   && max_steps > 0;
        }
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
    bool transporter_result{true};  //!< Output transporter result event data
    bool status_checker{false};  //!< Detailed debug checking of track states
    size_type log_progress{1};  //!< CELER_LOG progress every N events

    // Control
    unsigned int seed{};
    size_type num_track_slots{};  //!< Divided among streams. Defaults to 2^20
                                  //!< on device, 2^12 on host
    size_type initializer_capacity{};  //!< Divided among streams. Defaults to
                                       //!< 8 * num_track_slots
    size_type max_steps = static_cast<size_type>(-1);  //!< Step *iterations*
    InterpolationType interpolation{InterpolationType::linear};
    size_type poly_spline_order{1};
    real_type secondary_stack_factor{2};
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
               && log_progress > 0 && (field == no_field() || field_options);
    }
};

//---------------------------------------------------------------------------//
// Convert to standalone input format
inp::StandaloneInput to_input(RunnerInput const&);

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
