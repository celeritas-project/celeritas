//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-loop/LDemoIO.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>
#include <nlohmann/json.hpp>

#include "corecel/Types.hh"
#include "corecel/math/NumericLimits.hh"
#include "celeritas/ext/GeantSetup.hh"

#include "Transporter.hh"

namespace demo_loop
{
//---------------------------------------------------------------------------//
/*!
 * Input for a single run.
 */
struct LDemoArgs
{
    using real_type = celeritas::real_type;
    using size_type = celeritas::size_type;

    // Problem definition
    std::string geometry_filename; //!< Path to GDML file
    std::string physics_filename;  //!< Path to ROOT exported Geant4 data
    std::string hepmc3_filename;   //!< Path to Hepmc3 event data

    // Control
    unsigned int seed{};
    size_type    max_num_tracks{};
    size_type    max_steps = TransporterInput::no_max_steps();
    size_type    initializer_capacity{};
    real_type    secondary_stack_factor{};
    bool         enable_diagnostics{};
    bool         use_device{};
    bool         sync{};

    // Optional fixed-size step limiter for charged particles
    // (non-positive for unused)
    real_type step_limiter{};

    // Options for physics
    bool rayleigh{true};
    bool eloss_fluctuation{true};
    bool brem_combined{true};
    bool brem_lpm{true};
    bool conv_lpm{true};
    bool enable_msc{false};

    // Diagnostic input
    EnergyDiagInput energy_diag;

    // Optional setup options if loading directly from Geant4
    celeritas::GeantSetupOptions geant_options;

    //! Whether the run arguments are valid
    explicit operator bool() const
    {
        return !geometry_filename.empty() && !physics_filename.empty()
               && !hepmc3_filename.empty() && max_num_tracks > 0
               && max_steps > 0 && initializer_capacity > 0
               && secondary_stack_factor > 0;
    }
};

// Load params from input arguments
TransporterInput load_input(const LDemoArgs& args);

// Build transporter from input arguments
std::unique_ptr<TransporterBase> build_transporter(const LDemoArgs& run_args);

void to_json(nlohmann::json& j, const LDemoArgs& value);
void from_json(const nlohmann::json& j, LDemoArgs& value);

//---------------------------------------------------------------------------//
} // namespace demo_loop
