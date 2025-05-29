//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ddceler/DDcelerTMI.cc
//---------------------------------------------------------------------------//
#include "DDcelerTMI.hh"

#include <DDG4/Factories.h>
#include <QGSP_BERT.hh>

using TMI = celeritas::TrackingManagerIntegration;

namespace dd4hep
{
namespace sim
{
//---------------------------------------------------------------------------//
celeritas::SetupOptions DDcelerTMI::makeOptions()
{
    celeritas::SetupOptions opts;
    // NOTE: these numbers are appropriate for CPU execution and can be set
    // through the UI using `/celer/`
    opts.max_num_tracks = 2024;
    opts.initializer_capacity = 2024 * 128;
    // Celeritas does not support EmStandard MSC physics above 200 MeV
    opts.ignore_processes = {"CoulombScat"};

    // Use a uniform (zero) magnetic field
    opts.make_along_step = celeritas::UniformAlongStepFactory();

    // Save diagnostic file to a unique name
    opts.output_file = "trackingmanager-offload.out.json";
    return opts;
}
//---------------------------------------------------------------------------//

void DDcelerTMI::constructPhysics(G4VModularPhysicsList* physics)
{
    this->info(
        "Using FTFP_BERT physics list with Celeritas tracking for "
        "e-/e+/gamma.");

    // Register Celeritas tracking manager
    auto& tmi = TMI::Instance();
    physics->RegisterPhysics(new celeritas::TrackingManagerConstructor(&tmi));

    // Configure Celeritas options
    tmi.SetOptions(makeOptions());

    this->info("Celeritas TrackingManager registered.");
}
//---------------------------------------------------------------------------//
}  // namespace sim
}  // namespace dd4hep

DECLARE_GEANT4ACTION(DDcelerTMI)
