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
    // Celeritas does not support EmStandard MSC physics above 200 MeV
    opts.ignore_processes = {"CoulombScat"};

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

    // Strategy 1: Check if all components are ConstantField (fast path)
    Direction summed_direction(0, 0, 0);
    bool all_constant = true;

    for (auto const& mag_component : overlayed_obj->magnetic_components)
    {
        auto* cartesian_obj = mag_component.data<CartesianField::Object>();
        auto* const_field = dynamic_cast<ConstantField const*>(cartesian_obj);

        if (!const_field)
        {
            all_constant = false;
            break;
        }

        summed_direction
            = Direction(summed_direction.X() + const_field->direction.X(),
                        summed_direction.Y() + const_field->direction.Y(),
                        summed_direction.Z() + const_field->direction.Z());
    }

    Direction field_direction;

    if (all_constant)
    {
        this->info(("All "
                    + std::to_string(overlayed_obj->magnetic_components.size())
                    + " magnetic component(s) are ConstantField.")
                       .c_str());
        field_direction = summed_direction;
    }
    else
    {
        // Strategy 2: Sample the combined field to check if it's uniform
        this->info(
            "Non-constant field component(s) detected. Sampling combined "
            "field to verify uniformity...");

        constexpr double tolerance = 1e-6;  // tolerance in tesla
        constexpr size_t num_samples = 20;
        constexpr double sample_range_xy = 100.0;  // cm
        constexpr double sample_range_z = 200.0;  // cm

        // Generate random sampling points
        std::vector<Position> sample_points;
        sample_points.reserve(num_samples);

        // Always include origin
        sample_points.push_back(Position(0, 0, 0));

        // Add random points
        std::srand(12345);  // Fixed seed for reproducibility
        for (size_t i = 1; i < num_samples; ++i)
        {
            double x = (std::rand() / double(RAND_MAX) * 2.0 - 1.0)
                       * sample_range_xy;
            double y = (std::rand() / double(RAND_MAX) * 2.0 - 1.0)
                       * sample_range_xy;
            double z = (std::rand() / double(RAND_MAX) * 2.0 - 1.0)
                       * sample_range_z;
            sample_points.push_back(Position(x, y, z));
        }

        Direction reference = field.magneticField(sample_points[0]);
        bool is_uniform = true;

        for (size_t i = 1; i < sample_points.size(); ++i)
        {
            Direction sampled = field.magneticField(sample_points[i]);

            double dx = std::abs(sampled.X() - reference.X());
            double dy = std::abs(sampled.Y() - reference.Y());
            double dz = std::abs(sampled.Z() - reference.Z());

            // Compare in DD4hep units (already in tesla)
            if (dx > tolerance * dd4hep::tesla || dy > tolerance * dd4hep::tesla
                || dz > tolerance * dd4hep::tesla)
            {
                is_uniform = false;
                this->warning(("Field non-uniformity detected at position ("
                               + std::to_string(sample_points[i].X()) + ", "
                               + std::to_string(sample_points[i].Y()) + ", "
                               + std::to_string(sample_points[i].Z())
                               + ") cm: delta = ("
                               + std::to_string(dx / dd4hep::tesla) + ", "
                               + std::to_string(dy / dd4hep::tesla) + ", "
                               + std::to_string(dz / dd4hep::tesla) + ") T")
                                  .c_str());
                break;
            }
        }

        if (!is_uniform)
        {
            throw std::runtime_error(
                "Celeritas currently only supports uniform magnetic fields. "
                "The combined field is non-uniform (sampled at "
                + std::to_string(num_samples) + " random points).");
        }

        this->info(("Combined field verified to be uniform across "
                    + std::to_string(num_samples) + " random sample points.")
                       .c_str());
        field_direction = reference;
    }

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
