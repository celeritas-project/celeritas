//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/RunInput.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>

#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/sys/Environment.hh"
#include "celeritas/ext/GeantPhysicsOptions.hh"
#include "celeritas/field/FieldDriverOptions.hh"
#include "celeritas/phys/PrimaryGeneratorOptions.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
//! Physics list selection
enum class PhysicsListSelection
{
    ftfp_bert,
    geant_physics_list,
    size_,
};

//---------------------------------------------------------------------------//
/*!
 * Input for a single run.
 *
 * \todo Unify names with celer-sim and SetupOptions.
 */
struct RunInput
{
    using Real3 = Array<real_type, 3>;

    static constexpr Real3 no_field() { return Real3{0, 0, 0}; }
    static constexpr size_type unspecified{static_cast<size_type>(-1)};

    // Problem definition
    std::string geometry_file;  //!< Path to GDML file
    std::string event_file;  //!< Path to HepMC3 event record file

    // Setup options for generating primaries from a distribution
    PrimaryGeneratorOptions primary_options;

    // Control
    size_type num_track_slots{};
    size_type max_events{};
    size_type max_steps{unspecified};
    size_type initializer_capacity{};
    real_type secondary_stack_factor{};
    size_type cuda_stack_size{};
    size_type cuda_heap_size{};
    bool sync{false};
    bool default_stream{false};  //!< Launch all kernels on the default stream

    // Physics setup options
    PhysicsListSelection physics_list{PhysicsListSelection::ftfp_bert};
    GeantPhysicsOptions physics_options;

    // Field setup options
    std::string field_type{"uniform"};
    std::string field_file;
    Real3 field{no_field()};
    FieldDriverOptions field_options;

    // Sensitive detector hit collection
    bool enable_sd{true};

    // IO
    std::string output_file;  //!< Save JSON diagnostics
    std::string physics_output_file;  //!< Save physics data
    std::string offload_output_file;  //!< Save offloaded tracks to HepMC3/ROOT
    std::string macro_file;  //!< Load additional Geant4 commands
    int root_buffer_size{128000};
    bool write_sd_hits{false};
    bool strip_gdml_pointers{true};

    // Geant4 diagnostics
    bool step_diagnostic{false};
    int step_diagnostic_bins{1000};

    //! Whether the run arguments are valid
    explicit operator bool() const
    {
        return !geometry_file.empty()
               && (primary_options || !event_file.empty())
               && physics_list < PhysicsListSelection::size_
               && (field == no_field() || field_options)
               && ((num_track_slots > 0 && max_events > 0 && max_steps > 0
                    && initializer_capacity > 0 && secondary_stack_factor > 0)
                   || !celeritas::getenv("CELER_DISABLE").empty())
               && (step_diagnostic_bins > 0 || !step_diagnostic);
    }
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

char const* to_cstring(PhysicsListSelection value);

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
