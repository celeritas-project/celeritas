//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LDemoIO.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>
#include <nlohmann/json.hpp>
#include "base/NumericLimits.hh"
#include "base/Types.hh"
#include "sim/TrackInitParams.hh"
#include "Transporter.hh"

namespace celeritas
{
class ParticleParams;
}

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
    size_type    max_steps = celeritas::numeric_limits<size_type>::max();
    size_type    initializer_capacity{};
    real_type    secondary_stack_factor{};
    bool         enable_diagnostics{};
    bool         use_device{};
    bool         sync{};

    // Options for physics processes and models
    bool combined_brem{true};
    bool enable_lpm{true};

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
celeritas::TransporterInput load_input(const LDemoArgs& args);
std::shared_ptr<celeritas::TrackInitParams>

// Load primary particles from an input HepMC3 event file
load_primaries(const std::shared_ptr<const celeritas::ParticleParams>& particles,
               const LDemoArgs&                                        args);

// Build transporter from input arguments
std::unique_ptr<celeritas::TransporterBase>
build_transporter(const LDemoArgs& run_args);

void to_json(nlohmann::json& j, const LDemoArgs& value);
void from_json(const nlohmann::json& j, LDemoArgs& value);

//---------------------------------------------------------------------------//
} // namespace demo_loop
