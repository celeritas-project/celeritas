//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/RunInput.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>

#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/grid/GridTypes.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/Environment.hh"
#include "celeritas/ext/GeantPhysicsOptions.hh"
#include "celeritas/field/FieldDriverOptions.hh"
#include "celeritas/phys/PrimaryGeneratorOptions.hh"

#ifdef _WIN32
#    include <cstdlib>
#    ifdef environ
#        undef environ
#    endif
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
namespace inp
{
struct StandaloneInput;
}

namespace app
{
//---------------------------------------------------------------------------//
//! Sensitive detector capability
enum class SensitiveDetectorType
{
    none,  //!< No SDs
    simple_calo,  //!< Integrated energy deposition over all events
    event_hit,  //!< Record basic hit data
    size_,
};

//---------------------------------------------------------------------------//
//! Physics list selection (TODO: remove)
enum class PhysicsListSelection
{
    ftfp_bert,
    celer_ftfp_bert,  //!< FTFP BERT with Celeritas EM standard physics
    celer_em,  //!< Celeritas EM standard physics only
    size_,
};

//---------------------------------------------------------------------------//
/*!
 * Input for a single run.
 */
struct RunInput
{
    using Real3 = Array<real_type, 3>;

    static constexpr Real3 no_field() { return Real3{0, 0, 0}; }
    static constexpr size_type unspecified{static_cast<size_type>(-1)};

    // Global environment
    size_type cuda_stack_size{unspecified};
    size_type cuda_heap_size{unspecified};
    Environment environ;  //!< Supplement existing env variables

    // Problem definition
    std::string geometry_file;  //!< Path to GDML file
    std::string event_file;  //!< Path to HepMC3 event record file

    // Setup options for generating primaries from a distribution
    PrimaryGeneratorOptions primary_options;

    // Control
    size_type num_track_slots{};
    size_type max_steps{unspecified};
    size_type initializer_capacity{};
    real_type secondary_stack_factor{};
    size_type auto_flush{};  //!< Defaults to num_track_slots

    bool action_times{false};

    // Track reordering options
    TrackOrder track_order{Device::num_devices() ? TrackOrder::init_charge
                                                 : TrackOrder::none};

    // Physics setup options
    PhysicsListSelection physics_list{PhysicsListSelection::celer_ftfp_bert};
    GeantPhysicsOptions physics_options;
    InterpolationType interpolation{InterpolationType::linear};
    size_type poly_spline_order{1};

    // Field setup options
    std::string field_type{"uniform"};
    std::string field_file;
    Real3 field{no_field()};  //!< Field vector [T]
    FieldDriverOptions field_options;

    // SD setup options
    SensitiveDetectorType sd_type{SensitiveDetectorType::event_hit};

    // IO
    std::string output_file;  //!< Save JSON diagnostics
    std::string physics_output_file;  //!< Save physics data
    std::string offload_output_file;  //!< Save offloaded tracks to HepMC3/ROOT
    std::string macro_file;  //!< Load additional Geant4 commands

    // Geant4 diagnostics
    bool step_diagnostic{false};
    int step_diagnostic_bins{1000};
    std::string slot_diagnostic_prefix;
    std::string tracing_file;

    // Whether the run arguments are valid
    explicit operator bool() const;
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

char const* to_cstring(PhysicsListSelection value);
char const* to_cstring(SensitiveDetectorType value);

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
