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

#include "corecel/Config.hh"

#include "corecel/io/Logger.hh"
#include "celeritas/field/FieldDriverOptions.hh"
#include "celeritas/inp/Field.hh"
#include "accel/TrackingManagerIntegration.hh"

#if CELERITAS_USE_COVFIE
#    include "celeritas/field/CartMapFieldInput.hh"
#    include "LoadCovfieField.hh"
#endif

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
    declareProperty("FieldMapFile", field_map_file_);
}

//---------------------------------------------------------------------------//
SetupOptions CelerPhysics::make_options()
{
    SetupOptions opts;

    // Validate configuration parameters
    CELER_VALIDATE(max_num_tracks_ > 0,
                   << "invalid MaxNumTracks=" << max_num_tracks_
                   << "(should be positive)");
    CELER_VALIDATE(init_capacity_ > 0,
                   << "invalid InitCapacity=" << init_capacity_
                   << " (should be positive)");

    opts.max_num_tracks = max_num_tracks_;
    opts.initializer_capacity = init_capacity_;

    // Set ignored processes from configuration
    for (auto const& proc : ignore_processes_)
    {
        opts.ignore_processes.push_back(proc);
    }

    // Load field driver options from the DD4hep MagFieldTrackingSetup action
    dd4hep::sim::Geant4Action* field_action = nullptr;
    if (auto* config_phase = context()->kernel().getPhase("configure"))
    {
        for (auto const& [action, callback] : config_phase->members())
        {
            if (action->name() == "MagFieldTrackingSetup")
            {
                field_action = action;
                break;
            }
        }
    }

    FieldDriverOptions driver_options;
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

    CELER_VALIDATE(field_map_file_.empty() || CELERITAS_USE_COVFIE,
                   << "FieldMapFile='" << field_map_file_
                   << "' was set but Celeritas was built without covfie "
                      "support (CELERITAS_USE_covfie=OFF)");

#if CELERITAS_USE_COVFIE
    if (!field_map_file_.empty())
    {
        // Covfie field map mode: load binary field file
        CELER_LOG(info) << "Loading covfie field map from '" << field_map_file_
                        << "'";

        auto load_field = [filename = field_map_file_, driver_options] {
            CartMapFieldInput inp = LoadCovfieField(filename);
            inp.driver_options = driver_options;
            return inp;
        };
        opts.make_along_step = CartMapFieldAlongStepFactory(load_field);

        CELER_LOG(info) << "Using covfie CartMapField for along-step";
    }
    else
#endif
    {
        // Uniform field mode: read ConstantField from DD4hep detector
        // description
        auto& detector = context()->detectorDescription();
        auto&& field = detector.field();
        auto* overlaid_obj = field.data<OverlayedField::Object>();

        CELER_VALIDATE(overlaid_obj->electric_components.empty(),
                       << "Celeritas does not support electric field "
                          "components. Found "
                       << overlaid_obj->electric_components.size()
                       << " electric component(s).");

        CELER_VALIDATE(!overlaid_obj->magnetic_components.empty(),
                       << "No magnetic field components found in DD4hep field "
                          "description.");

        // Sum all ConstantField components
        Direction field_direction(0, 0, 0);
        for (auto const& mag_component : overlaid_obj->magnetic_components)
        {
            auto* cartesian_obj
                = mag_component.data<CartesianField::Object>();
            auto* const_field
                = dynamic_cast<ConstantField const*>(cartesian_obj);

            CELER_VALIDATE(
                const_field,
                << "Celeritas uniform field mode only supports ConstantField "
                   "components. Found non-constant field in DD4hep description."
                   " Set FieldMapFile to use a covfie field map instead.");
            field_direction += const_field->direction;
        }

        constexpr double dd4hep_tesla = dd4hep::tesla;
        CELER_LOG(debug) << "Field strength: ("
                         << field_direction.X() / dd4hep_tesla << ", "
                         << field_direction.Y() / dd4hep_tesla << ", "
                         << field_direction.Z() / dd4hep_tesla << ") T";

        auto make_field_input = [field_direction, driver_options] {
            inp::UniformField input;
            constexpr double dd4hep_t = dd4hep::tesla;
            input.strength = {field_direction.X() / dd4hep_t,
                              field_direction.Y() / dd4hep_t,
                              field_direction.Z() / dd4hep_t};
            input.driver_options = driver_options;
            return input;
        };
        opts.make_along_step = UniformAlongStepFactory(make_field_input);
    }
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
