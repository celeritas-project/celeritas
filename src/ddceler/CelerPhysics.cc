//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ddceler/CelerPhysics.cc
//---------------------------------------------------------------------------//
#include "CelerPhysics.hh"

#include <DD4hep/Detector.h>
#include <DD4hep/FieldTypes.h>
#include <DDG4/Factories.h>
#include <DDG4/Geant4ActionPhase.h>
#include <DDG4/Geant4Kernel.h>

#include "celeritas/field/FieldDriverOptions.hh"
#include "celeritas/inp/Field.hh"
#include "accel/TrackingManagerIntegration.hh"

using TMI = celeritas::TrackingManagerIntegration;
using Geant4Context = dd4hep::sim::Geant4Context;
using Geant4PhysicsList = dd4hep::sim::Geant4PhysicsList;
using OverlayedField = dd4hep::OverlayedField;
using CartesianField = dd4hep::CartesianField;
using ConstantField = dd4hep::ConstantField;
using Direction = dd4hep::Direction;

namespace celeritas
{
namespace dd
{
namespace
{
//---------------------------------------------------------------------------//

FieldDriverOptions load_driver_options(dd4hep::sim::Geant4Action* field_action)
{
    FieldDriverOptions driver_options;
    constexpr auto celer_mm = units::millimeter;

    // Load field tracking parameters directly from DD4hep action properties
    // Values are in DD4hep units (mm)
    driver_options.delta_chord
        = field_action->property("delta_chord").value<double>() * celer_mm;
    driver_options.delta_intersection
        = field_action->property("delta_intersection").value<double>()
          * celer_mm;
    driver_options.minimum_step
        = field_action->property("delta_one_step").value<double>() * celer_mm;

    return driver_options;
}

}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Standard constructor
 */
CelerPhysics::CelerPhysics(Geant4Context* ctxt, std::string const& name)
    : Geant4PhysicsList(ctxt, name)
{
    declareProperty("MaxNumTracks", max_num_tracks_);
    declareProperty("InitCapacity", init_capacity_);
    declareProperty("IgnoreProcesses", ignore_processes_);
}

//---------------------------------------------------------------------------//
SetupOptions CelerPhysics::make_options()
{
    SetupOptions opts;

    // Validate configuration parameters
    CELER_VALIDATE(max_num_tracks_ > 0,
                   << "MaxNumTracks must be set to a positive value (got "
                   << max_num_tracks_ << ")");
    CELER_VALIDATE(init_capacity_ > 0,
                   << "InitCapacity must be set to a positive value (got "
                   << init_capacity_ << ")");

    opts.max_num_tracks = max_num_tracks_;
    opts.initializer_capacity = init_capacity_;

    // Set ignored processes from configuration
    for (auto const& proc : ignore_processes_)
    {
        opts.ignore_processes.push_back(proc);
    }

    // Get the field from DD4hep detector description and validate its type
    auto& detector = context()->detectorDescription();
    auto field = detector.field();
    auto* overlaid_obj = field.data<OverlayedField::Object>();

    // Validate field configuration: no electric components
    CELER_VALIDATE(overlaid_obj->electric_components.empty(),
                   << "Celeritas does not support electric field components. "
                      "Found "
                   << overlaid_obj->electric_components.size()
                   << " electric component(s).");

    CELER_VALIDATE(!overlaid_obj->magnetic_components.empty(),
                   << "No magnetic field components found in DD4hep field "
                      "description.");

    // Check that all magnetic components are ConstantField and sum them
    Direction field_direction(0, 0, 0);
    for (auto const& mag_component : overlaid_obj->magnetic_components)
    {
        auto* cartesian_obj = mag_component.data<CartesianField::Object>();
        auto* const_field = dynamic_cast<ConstantField const*>(cartesian_obj);

        CELER_VALIDATE(const_field,
                       << "Celeritas currently only supports ConstantField "
                          "magnetic "
                       << "fields. Found non-constant field component in "
                          "DD4hep "
                       << "description.");
        field_direction += const_field->direction;
    }

    // Print field strength
    // Note: field_direction is already in DD4hep internal units (parsed from
    // XML) DD4hep supports tesla, gauss, kilogauss, etc. in XML and converts
    // to internal units
    constexpr double dd4hep_tesla = dd4hep::tesla;
    CELER_LOG(debug) << "Field strength: ("
                     << field_direction.X() / dd4hep_tesla << ", "
                     << field_direction.Y() / dd4hep_tesla << ", "
                     << field_direction.Z() / dd4hep_tesla << ") T";

    // Get field tracking parameters from DD4hep FieldSetup action
    // These parameters are set in the steering file (runner.field.*)
    FieldDriverOptions driver_options;

    auto& kernel = context()->kernel();
    auto* config_phase = kernel.getPhase("configure");

    dd4hep::sim::Geant4Action* field_action = nullptr;
    if (config_phase)
    {
        // Find the MagFieldTrackingSetup action in the configure phase
        for (auto const& [action, callback] : config_phase->members())
        {
            if (action->name() == "MagFieldTrackingSetup")
            {
                field_action = action;
                break;
            }
        }
    }

    if (field_action)
    {
        driver_options = load_driver_options(field_action);
        CELER_LOG(debug) << "Loaded field driver options from DD4hep "
                            "FieldSetup action";
    }
    else
    {
        CELER_LOG(warning) << "MagFieldTrackingSetup action not found, using "
                              "default field parameters";
    }

    // Print field driver options
    constexpr auto celer_mm = units::millimeter;
    CELER_LOG(debug)
        << "Field driver options: min_step="
        << driver_options.minimum_step / celer_mm
        << " mm, delta_chord=" << driver_options.delta_chord / celer_mm
        << " mm, delta_intersection="
        << driver_options.delta_intersection / celer_mm << " mm";

    // Use a uniform magnetic field based on DD4hep ConstantField
    auto make_field_input = [field_direction, driver_options] {
        inp::UniformField input;

        // Convert from DD4hep (tesla) to Celeritas field units
        input.strength = {field_direction.X() / dd4hep_tesla,
                          field_direction.Y() / dd4hep_tesla,
                          field_direction.Z() / dd4hep_tesla};
        input.driver_options = driver_options;
        return input;
    };
    opts.make_along_step = UniformAlongStepFactory(make_field_input);
    opts.sd.ignore_zero_deposition = false;

    // Save diagnostic file to a unique name
    opts.output_file = "ddceler.out.json";
    opts.geometry_output_file = "ddceler.out.gdml";
    return opts;
}

//---------------------------------------------------------------------------//

void CelerPhysics::constructPhysics(G4VModularPhysicsList* physics)
{
    // Register Celeritas tracking manager
    auto& tmi = TMI::Instance();
    physics->RegisterPhysics(new TrackingManagerConstructor(&tmi));

    // Configure Celeritas options
    tmi.SetOptions(this->make_options());
}

//---------------------------------------------------------------------------//
}  // namespace dd
}  // namespace celeritas

DECLARE_GEANT4ACTION_NS(celeritas::dd, CelerPhysics)
