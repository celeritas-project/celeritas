//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/setup/FrameworkInput.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "Problem.hh"

namespace celeritas
{
class CoreParams;
class GeantGeoParams;

namespace inp
{
struct FrameworkInput;
}
namespace setup
{
//---------------------------------------------------------------------------//
//! Result from loaded standalone input to be used in front-end apps
struct FrameworkLoaded
{
    //! Loaded problem
    ProblemLoaded problem;
    //! Geant4 geometry wrapper
    std::shared_ptr<GeantGeoParams> geo;
};

//---------------------------------------------------------------------------//
// Completely set up a Celeritas problem from a framework input
FrameworkLoaded framework_input(inp::FrameworkInput& fi);

//---------------------------------------------------------------------------//
}  // namespace setup
}  // namespace celeritas
