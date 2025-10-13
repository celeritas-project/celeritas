//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ddceler/DDcelerTMI.cc
//---------------------------------------------------------------------------//
#include "DDcelerTMI.hh"

#include <DD4hep/Detector.h>
#include <DDG4/Factories.h>

#include "celeritas/field/FieldDriverOptions.hh"
#include "celeritas/inp/Field.hh"

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
    opts.max_num_tracks = m_maxNumTracks;
    opts.initializer_capacity = m_initCapacity;
    // Celeritas does not support EmStandard MSC physics above 200 MeV
    opts.ignore_processes = {"CoulombScat"};

    // Get the field from DD4hep detector description
    auto& detector = context()->detectorDescription();
    auto field = detector.field();
    auto uniform_field_strength = field.magneticField(Position(0, 0, 0));

    this->info(
        "Celeritas will assume a constant field everywhere that matches "
        "the field value at origin as specified in dd4hep xml description.");

    // Use a uniform magnetic field based on DD4hep field at origin
    auto make_field_input = [uniform_field_strength] {
        celeritas::inp::UniformField input;

        // Convert from DD4hep units (tesla) to Celeritas field units
        constexpr double dd4hep_tesla = dd4hep::tesla;
        input.strength = {uniform_field_strength.X() / dd4hep_tesla,
                          uniform_field_strength.Y() / dd4hep_tesla,
                          uniform_field_strength.Z() / dd4hep_tesla};

        constexpr auto celer_mm = celeritas::units::millimeter;
        input.driver_options.minimum_step = 1e-6 * celer_mm;
        input.driver_options.delta_chord = 0.025 * celer_mm;
        input.driver_options.delta_intersection = 1e-5 * celer_mm;
        return input;
    };
    opts.make_along_step = celeritas::UniformAlongStepFactory(make_field_input);
    opts.sd.ignore_zero_deposition = false;

    // Save diagnostic file to a unique name
    opts.output_file = "trackingmanager-offload.out.json";
    return opts;
}
//---------------------------------------------------------------------------//

void DDcelerTMI::constructPhysics(G4VModularPhysicsList* physics)
{
    this->info(
        "Using Celeritas tracking for "
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
