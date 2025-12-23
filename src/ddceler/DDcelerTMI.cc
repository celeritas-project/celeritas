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
#include <Evaluator/DD4hepUnits.h>
#include <Evaluator/Evaluator.h>
#include <G4FieldManager.hh>
#include <G4TransportationManager.hh>

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
namespace ddceler
{

//---------------------------------------------------------------------------//
FieldDriverOptions
load_driver_options(std::map<std::string, std::string> const& tracking_props)
{
    FieldDriverOptions result;
    // Create evaluator for parsing expressions with units
    dd4hep::tools::Evaluator eval;

    auto set_param
        = [&tracking_props, &eval](std::string const& key, double& val) {
              if (!tracking_props.count(key))
                  return;

              // Values from RUNNER.field can include units (e.g., "0.025*mm"
              // or "0.025")
              std::string const& value_str = tracking_props.at(key);

              // Evaluate expression to get value in DD4hep internal units
              // (mm=1)
              auto eval_result = eval.evaluate(value_str.c_str());
              CELER_VALIDATE(eval_result.first == dd4hep::tools::Evaluator::OK,
                             << "failed to parse field tracking parameter '"
                             << key << "' with value '" << value_str << "'");

              // eval_result.second is already in mm (DD4hep's base unit)
              // Convert to Celeritas internal units
              constexpr auto celer_mm = units::millimeter;
              val = eval_result.second * celer_mm;
          };

    set_param("min_chord_step", result.minimum_step);
    set_param("delta_chord", result.delta_chord);
    set_param("delta_intersection", result.delta_intersection);

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Standard constructor
 */
DDcelerTMI::DDcelerTMI(Geant4Context* ctxt, std::string const& name)
    : Geant4PhysicsList(ctxt, name)
{
    declareProperty("MaxNumTracks", max_num_tracks_);
    declareProperty("InitCapacity", init_capacity_);
    declareProperty("IgnoreProcesses", ignore_processes_);
}

//---------------------------------------------------------------------------//
SetupOptions DDcelerTMI::make_options()
{
    SetupOptions opts;

    // Validate configuration parameters
    CELER_VALIDATE(max_num_tracks_ > 0,
                   << "MaxNumTracks must be set to a positive value (got "
                   << max_num_tracks_ << ")");
    CELER_VALIDATE(init_capacity_ > 0,
                   << "InitCapacity must be set to a positive value (got "
                   << init_capacity_ << ")");

    // NOTE: these numbers are appropriate for CPU execution and can be set
    // through the UI using `/celer/`
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
    Direction summed_direction(0, 0, 0);

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

        summed_direction
            = Direction(summed_direction.X() + const_field->direction.X(),
                        summed_direction.Y() + const_field->direction.Y(),
                        summed_direction.Z() + const_field->direction.Z());
    }

    this->info(("All "
                + std::to_string(overlaid_obj->magnetic_components.size())
                + " magnetic component(s) are ConstantField.")
                   .c_str());

    Direction field_direction = summed_direction;

    // Print field strength
    // Note: field_direction is already in DD4hep internal units (parsed from
    // XML) DD4hep supports tesla, gauss, kilogauss, etc. in XML and converts
    // to internal units
    constexpr double dd4hep_tesla = dd4hep::tesla;
    CELER_LOG(debug) << "Field strength: ("
                     << field_direction.X() / dd4hep_tesla << ", "
                     << field_direction.Y() / dd4hep_tesla << ", "
                     << field_direction.Z() / dd4hep_tesla << ") T";

    // Query field tracking parameters from DD4hep overlaid field properties
    // These are set in the steering file via RUNNER.field.*
    FieldDriverOptions driver_options;
    auto const& overlaid_properties = overlaid_obj->properties;
    if (auto iter = overlaid_properties.find("field_tracking");
        iter != overlaid_properties.end())
    {
        driver_options = load_driver_options(iter->second);
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
    return opts;
}

//---------------------------------------------------------------------------//

void DDcelerTMI::constructPhysics(G4VModularPhysicsList* physics)
{
    // Register Celeritas tracking manager
    auto& tmi = TMI::Instance();
    physics->RegisterPhysics(new TrackingManagerConstructor(&tmi));

    // Configure Celeritas options
    tmi.SetOptions(this->make_options());
}

//---------------------------------------------------------------------------//
}  // namespace ddceler
}  // namespace celeritas

DECLARE_GEANT4ACTION_NS(celeritas::ddceler, DDcelerTMI)
