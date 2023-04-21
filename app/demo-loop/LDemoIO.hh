//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-loop/LDemoIO.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>
#include <celeritas/Types.hh>
#include <nlohmann/json.hpp>

#include "celeritas_config.h"
#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/sys/Environment.hh"
#include "celeritas/ext/GeantPhysicsOptions.hh"
#include "celeritas/ext/GeantSetup.hh"
#include "celeritas/ext/RootFileManager.hh"
#include "celeritas/field/FieldDriverOptions.hh"
#include "celeritas/phys/PrimaryGeneratorOptions.hh"
#include "celeritas/user/RootStepWriter.hh"

#include "Transporter.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class OutputRegistry;
class CoreParams;
//---------------------------------------------------------------------------//
}

namespace demo_loop
{
//---------------------------------------------------------------------------//
/*!
 * Input for a single run.
 *
 * TODO: change to RunnerInput and move to Runner.hh
 */
struct LDemoArgs
{
    using real_type = celeritas::real_type;
    using Real3 = celeritas::Real3;
    using size_type = celeritas::size_type;

    static constexpr Real3 no_field() { return Real3{0, 0, 0}; }
    static constexpr size_type unspecified{static_cast<size_type>(-1)};

    // Global environment
    size_type cuda_heap_size = unspecified;
    size_type cuda_stack_size = unspecified;
    celeritas::Environment environ;  //!< Supplement existing env variables

    // Problem definition
    std::string geometry_filename;  //!< Path to GDML file
    std::string physics_filename;  //!< Path to ROOT exported Geant4 data
    std::string hepmc3_filename;  //!< Path to HepMC3 event data
    std::string mctruth_filename;  //!< Path to ROOT MC truth event data

    // Optional filter for ROOT MC truth data
    celeritas::SimpleRootFilterInput mctruth_filter;

    // Optional setup options for generating primaries programmatically
    celeritas::PrimaryGeneratorOptions primary_gen_options;

    // Control
    unsigned int seed{};
    size_type max_num_tracks{};
    size_type max_steps = TransporterInput::no_max_steps();
    size_type initializer_capacity{};
    size_type max_events{};
    real_type secondary_stack_factor{};
    bool enable_diagnostics{};
    bool use_device{};
    bool sync{};

    // Magnetic field vector [* 1/Tesla] and associated field options
    Real3 mag_field{no_field()};
    celeritas::FieldDriverOptions field_options;

    // Optional fixed-size step limiter for charged particles
    // (non-positive for unused)
    real_type step_limiter{};

    // Options for physics
    bool brem_combined{true};

    // Track init options
    celeritas::TrackOrder track_order{celeritas::TrackOrder::unsorted};

    // Optional setup options if loading directly from Geant4
    celeritas::GeantPhysicsOptions geant_options;

    //! Whether the run arguments are valid
    explicit operator bool() const
    {
        return !geometry_filename.empty() && !physics_filename.empty()
               && (primary_gen_options || !hepmc3_filename.empty())
               && max_num_tracks > 0 && max_steps > 0
               && initializer_capacity > 0 && max_events > 0
               && secondary_stack_factor > 0
               && (mag_field == no_field() || field_options);
    }
};

// Build core params
std::shared_ptr<celeritas::CoreParams>
build_core_params(LDemoArgs const& args,
                  std::shared_ptr<celeritas::OutputRegistry> outreg);

// Build transporter from input arguments
std::unique_ptr<TransporterBase>
build_transporter(LDemoArgs const& args,
                  std::shared_ptr<celeritas::CoreParams const>);

void to_json(nlohmann::json& j, LDemoArgs const& value);
void from_json(nlohmann::json const& j, LDemoArgs& value);

// Store LDemoArgs to ROOT file when ROOT is available
void write_to_root(LDemoArgs const& cargs,
                   celeritas::RootFileManager* root_manager);

// Store CoreParams to ROOT file when ROOT is available
void write_to_root(celeritas::CoreParams const& core_params,
                   celeritas::RootFileManager* root_manager);

#if !CELERITAS_USE_ROOT
inline void write_to_root(LDemoArgs const&, celeritas::RootFileManager*)
{
    CELER_NOT_CONFIGURED("ROOT");
}

inline void
write_to_root(celeritas::CoreParams const&, celeritas::RootFileManager*)
{
    CELER_NOT_CONFIGURED("ROOT");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace demo_loop
