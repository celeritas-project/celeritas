//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/setup/StandaloneInput.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "celeritas/phys/Primary.hh"

#include "Problem.hh"

namespace celeritas
{
class CoreParams;
class GeantGeoParams;

namespace inp
{
struct OpticalStandaloneInput;
struct StandaloneInput;
}  // namespace inp
namespace setup
{
//---------------------------------------------------------------------------//
//! Result from loaded standalone input to be used in front-end apps
struct StandaloneLoaded
{
    using VecPrimary = std::vector<Primary>;
    using VecEvent = std::vector<VecPrimary>;

    //! Problem setup
    ProblemLoaded problem;
    //! Loaded Geant4 geometry (if inp.geant_setup)
    std::shared_ptr<GeantGeoParams> geant_geo;
    //! Events to be run
    VecEvent events;
};

//---------------------------------------------------------------------------//
//! Result from loaded optical-only standalone input for front-end apps
struct OpticalStandaloneLoaded
{
    //! Problem setup
    OpticalProblemLoaded problem;
    //! Loaded Geant4 geometry
    std::shared_ptr<GeantGeoParams> geant_geo;
};

//---------------------------------------------------------------------------//
// Completely set up a Celeritas problem from a standalone input
StandaloneLoaded standalone_input(inp::StandaloneInput& si);

// Completely set up a Celeritas optical-only problem from a standalone input
OpticalStandaloneLoaded standalone_input(inp::OpticalStandaloneInput& si);

//---------------------------------------------------------------------------//
}  // namespace setup
}  // namespace celeritas
