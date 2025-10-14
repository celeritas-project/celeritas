//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ddceler/DDcelerTMI.cc
//---------------------------------------------------------------------------//
#include "DDcelerTMI.hh"

#include <DD4hep/Detector.h>
#include <DD4hep/FieldTypes.h>
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

    // Set ignored processes from configuration
    for (auto const& proc : m_ignoreProcesses)
    {
        opts.ignore_processes.push_back(proc);
    }

    // Get the field from DD4hep detector description and validate its type
    auto& detector = context()->detectorDescription();
    auto field = detector.field();
    auto* overlayed_obj = field.data<OverlayedField::Object>();

    // Validate field configuration: no electric components
    if (!overlayed_obj->electric_components.empty())
    {
        throw std::runtime_error(
            "Celeritas does not support electric field components. Found "
            + std::to_string(overlayed_obj->electric_components.size())
            + " electric component(s).");
    }

    if (overlayed_obj->magnetic_components.empty())
    {
        throw std::runtime_error(
            "No magnetic field components found in DD4hep field description.");
    }

    // Check that all magnetic components are ConstantField and sum them
    Direction summed_direction(0, 0, 0);

    for (auto const& mag_component : overlayed_obj->magnetic_components)
    {
        auto* cartesian_obj = mag_component.data<CartesianField::Object>();
        auto* const_field = dynamic_cast<ConstantField const*>(cartesian_obj);

        if (!const_field)
        {
            throw std::runtime_error(
                "Celeritas currently only supports ConstantField magnetic "
                "fields. Found non-constant field component in DD4hep "
                "description.");
        }

        summed_direction
            = Direction(summed_direction.X() + const_field->direction.X(),
                        summed_direction.Y() + const_field->direction.Y(),
                        summed_direction.Z() + const_field->direction.Z());
    }

    this->info(("All "
                + std::to_string(overlayed_obj->magnetic_components.size())
                + " magnetic component(s) are ConstantField.")
                   .c_str());

    Direction field_direction = summed_direction;

    // Use a uniform magnetic field based on DD4hep ConstantField
    auto make_field_input = [field_direction] {
        celeritas::inp::UniformField input;

        // Convert from DD4hep units (tesla) to Celeritas field units
        constexpr double dd4hep_tesla = dd4hep::tesla;
        input.strength = {field_direction.X() / dd4hep_tesla,
                          field_direction.Y() / dd4hep_tesla,
                          field_direction.Z() / dd4hep_tesla};

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
