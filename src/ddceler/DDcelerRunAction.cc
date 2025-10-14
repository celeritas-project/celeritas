//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ddceler/DDcelerRunAction.cc
//---------------------------------------------------------------------------//
#include "DDcelerRunAction.hh"

#include <CeleritasG4.hh>
#include <DD4hep/Detector.h>
#include <DD4hep/FieldTypes.h>
#include <DDG4/Factories.h>
#include <Evaluator/DD4hepUnits.h>
#include <Evaluator/Evaluator.h>
#include <G4ChordFinder.hh>
#include <G4ClassicalRK4.hh>
#include <G4FieldManager.hh>
#include <G4Mag_UsualEqRhs.hh>
#include <G4MagneticField.hh>
#include <G4TransportationManager.hh>

#include "celeritas/Units.hh"

using TMI = celeritas::TrackingManagerIntegration;

namespace dd4hep
{
namespace sim
{

//---------------------------------------------------------------------------//

void DDcelerRunAction::begin(G4Run const* run)
{
    this->info("Begin of run");

    // Update Geant4 field tracking parameters
    // This runs after DD4hep's automatic field setup, so we can override the
    // parameters
    updateFieldTracking();

    TMI::Instance().BeginOfRunAction(run);
}

void DDcelerRunAction::updateFieldTracking()
{
    auto& detector = context()->detectorDescription();
    auto field = detector.field();
    auto* overlayed_obj = field.data<OverlayedField::Object>();
    auto const& overlayed_properties = overlayed_obj->properties;

    // Default values (in Celeritas units, then convert to DD4hep)
    constexpr auto celer_mm = celeritas::units::millimeter;
    constexpr auto dd4hep_mm = dd4hep::mm;

    double min_step = 1e-6 * celer_mm;
    double delta_chord = 0.025 * celer_mm;
    double delta_intersection = 1e-5 * celer_mm;
    double delta_one_step = 0.01 * celer_mm;
    double eps_min = 5e-5 * celer_mm;
    double eps_max = 0.001 * celer_mm;

    // Read from properties if available
    if (overlayed_properties.count("field_tracking"))
    {
        auto const& tracking_props = overlayed_properties.at("field_tracking");
        dd4hep::tools::Evaluator eval;

        auto parse_length = [&](std::string const& key) -> double {
            if (!tracking_props.count(key))
                return 0.0;
            auto result = eval.evaluate(tracking_props.at(key).c_str());
            if (result.first != dd4hep::tools::Evaluator::OK)
                return 0.0;
            return result.second / dd4hep::mm * celer_mm;
        };

        double val;
        if ((val = parse_length("min_chord_step")) > 0)
            min_step = val;
        if ((val = parse_length("delta_chord")) > 0)
            delta_chord = val;
        if ((val = parse_length("delta_intersection")) > 0)
            delta_intersection = val;
        if ((val = parse_length("delta_one_step")) > 0)
            delta_one_step = val;
        if ((val = parse_length("eps_min")) > 0)
            eps_min = val;
        if ((val = parse_length("eps_max")) > 0)
            eps_max = val;
    }

    // Convert to DD4hep units for Geant4
    double g4_min_chord_step = min_step / celer_mm * dd4hep_mm;
    double g4_delta_chord = delta_chord / celer_mm * dd4hep_mm;
    double g4_delta_intersection = delta_intersection / celer_mm * dd4hep_mm;
    double g4_delta_one_step = delta_one_step / celer_mm * dd4hep_mm;
    double g4_eps_min = eps_min / celer_mm * dd4hep_mm;
    double g4_eps_max = eps_max / celer_mm * dd4hep_mm;

    // Update field tracking
    auto* transport_mgr = G4TransportationManager::GetTransportationManager();
    auto* field_mgr = transport_mgr->GetFieldManager();
    auto* old_chord_finder = field_mgr->GetChordFinder();

    if (old_chord_finder)
    {
        auto* mag_field = const_cast<G4MagneticField*>(
            static_cast<G4MagneticField const*>(field_mgr->GetDetectorField()));

        auto* mag_equation = new G4Mag_UsualEqRhs(mag_field);
        auto* stepper = new G4ClassicalRK4(mag_equation);
        auto* new_chord_finder
            = new G4ChordFinder(mag_field, g4_min_chord_step, stepper);

        new_chord_finder->SetDeltaChord(g4_delta_chord);
        field_mgr->SetChordFinder(new_chord_finder);
        field_mgr->SetDeltaIntersection(g4_delta_intersection);
        field_mgr->SetDeltaOneStep(g4_delta_one_step);
        field_mgr->SetMinimumEpsilonStep(g4_eps_min);
        field_mgr->SetMaximumEpsilonStep(g4_eps_max);

        this->info("Geant4 field tracking parameters updated from XML.");
    }
    else
    {
        this->warning(
            "No ChordFinder found - cannot update field tracking parameters.");
    }
}

//---------------------------------------------------------------------------//

void DDcelerRunAction::end(G4Run const* run)
{
    this->info("End of run");
    TMI::Instance().EndOfRunAction(run);
}
//---------------------------------------------------------------------------//

}  // namespace sim
}  // namespace dd4hep

DECLARE_GEANT4ACTION(DDcelerRunAction)
