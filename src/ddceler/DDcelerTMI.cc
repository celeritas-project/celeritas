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
#include <DDG4/Geant4Field.h>
#include <Evaluator/DD4hepUnits.h>
#include <Evaluator/Evaluator.h>
#include <G4ChordFinder.hh>
#include <G4ClassicalRK4.hh>
#include <G4FieldManager.hh>
#include <G4MagIntegratorStepper.hh>
#include <G4Mag_UsualEqRhs.hh>
#include <G4TransportationManager.hh>

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

    // Print field strength
    // Note: field_direction is already in DD4hep internal units (parsed from
    // XML) DD4hep supports tesla, gauss, kilogauss, etc. in XML and converts
    // to internal units
    constexpr double dd4hep_tesla = dd4hep::tesla;
    CELER_LOG(debug) << "Field strength: ("
                     << field_direction.X() / dd4hep_tesla << ", "
                     << field_direction.Y() / dd4hep_tesla << ", "
                     << field_direction.Z() / dd4hep_tesla << ") T";

    // Query field properties for driver options from OverlayedField
    auto const& overlayed_properties = overlayed_obj->properties;

    // Default values for field driver options (in Celeritas units)
    // Note: Geant4-specific parameters (delta_one_step, eps_min, eps_max) are
    // handled in updateFieldTracking()
    constexpr auto celer_mm = celeritas::units::millimeter;
    double min_step = 1e-6 * celer_mm;
    double delta_chord = 0.025 * celer_mm;
    double delta_intersection = 1e-5 * celer_mm;

    // Try to read from DD4hep overlayed field properties if available
    if (overlayed_properties.count("field_tracking"))
    {
        auto const& tracking_props = overlayed_properties.at("field_tracking");

        // Create evaluator for parsing expressions with units
        // The evaluator uses DD4hep's default unit system
        dd4hep::tools::Evaluator eval;

        // Helper to parse property value with units using DD4hep's evaluator
        auto parse_length = [&](std::string const& key) -> double {
            if (!tracking_props.count(key))
                return 0.0;

            std::string const& value_str = tracking_props.at(key);

            // Evaluate expression (e.g., "1e-6*mm", "0.01*cm", etc.)
            // evaluate() returns pair<int, double> where first is status,
            // second is value
            auto result = eval.evaluate(value_str.c_str());

            if (result.first != dd4hep::tools::Evaluator::OK)
            {
                throw std::runtime_error("Failed to parse field property '" + key
                                         + "' with value '" + value_str + "'");
            }

            // Convert: DD4hep internal units -> Celeritas mm
            return result.second / dd4hep::mm * celer_mm;
        };

        if (tracking_props.count("min_chord_step"))
        {
            min_step = parse_length("min_chord_step");
        }
        if (tracking_props.count("delta_chord"))
        {
            delta_chord = parse_length("delta_chord");
        }
        if (tracking_props.count("delta_intersection"))
        {
            delta_intersection = parse_length("delta_intersection");
        }
    }

    // Print field driver options
    CELER_LOG(debug)
        << "Field driver options: min_step=" << min_step / celer_mm
        << " mm, delta_chord=" << delta_chord / celer_mm
        << " mm, delta_intersection=" << delta_intersection / celer_mm
        << " mm";

    // Use a uniform magnetic field based on DD4hep ConstantField
    auto make_field_input
        = [field_direction, min_step, delta_chord, delta_intersection] {
              celeritas::inp::UniformField input;

              // Convert from DD4hep units (tesla) to Celeritas field units
              constexpr double dd4hep_tesla = dd4hep::tesla;
              input.strength = {field_direction.X() / dd4hep_tesla,
                                field_direction.Y() / dd4hep_tesla,
                                field_direction.Z() / dd4hep_tesla};

              input.driver_options.minimum_step = min_step;
              input.driver_options.delta_chord = delta_chord;
              input.driver_options.delta_intersection = delta_intersection;
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
