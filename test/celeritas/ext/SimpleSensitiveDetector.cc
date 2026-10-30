//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/SimpleSensitiveDetector.cc
//---------------------------------------------------------------------------//
#include "SimpleSensitiveDetector.hh"

#include <iostream>
#include <CLHEP/Units/SystemOfUnits.h>
#include <G4LogicalVolume.hh>
#include <G4Step.hh>
#include <G4TouchableHistory.hh>
#include <G4Track.hh>

#include "corecel/Assert.hh"
#include "celeritas/ext/GeantStepPointView.hh"
#include "celeritas/ext/GeantStepView.hh"
#include "celeritas/ext/GeantTrackView.hh"

#include "TestMacros.hh"

using std::cout;

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
std::ostream& operator<<(std::ostream& os, SimpleHitsResult const& ref)
{
    return os << '{' << CELER_ATTR_REPR(ref, energy_deposition) << ", "
              << CELER_ATTR_REPR(ref, step_length) << ", "
              << CELER_ATTR_REPR(ref, particle) << ", "
              << CELER_ATTR_REPR(ref, weight) << ", "
              << CELER_ATTR_REPR(ref, track_id) << ", "
              << CELER_ATTR_REPR(ref, parent_id) << ", "
              << CELER_ATTR_REPR(ref, pre_energy) << ", "
              << CELER_ATTR_REPR(ref, pre_time) << ", "
              << CELER_ATTR_REPR(ref, pre_pos) << ", "
              << CELER_ATTR_REPR(ref, pre_dir) << ", "
              << CELER_ATTR_REPR(ref, pre_physvol) << ", "
              << CELER_ATTR_REPR(ref, post_time) << ", "
              << CELER_ATTR_REPR(ref, post_energy) << ", "
              << CELER_ATTR_REPR(ref, post_pos) << ", "
              << CELER_ATTR_REPR(ref, post_dir) << ", "
              << CELER_ATTR_REPR(ref, post_physvol) << ", "
              << CELER_ATTR_REPR(ref, post_status) << '}';
}

//---------------------------------------------------------------------------//
void check_ref_eq(AssertionHelper& helper,
                  SimpleHitsResult const& expected,
                  SimpleHitsResult const& actual)
{
    using namespace ::celeritas::test;
    is_vec_soft_eq(helper,
                   "energy_deposition",
                   expected.energy_deposition,
                   actual.energy_deposition);
    is_vec_soft_eq(
        helper, "step_length", expected.step_length, actual.step_length);
    is_vec_eq(helper, "particle", expected.particle, actual.particle);
    is_vec_soft_eq(helper, "weight", expected.weight, actual.weight);
    is_vec_eq(helper, "track_id", expected.track_id, actual.track_id);
    is_vec_eq(helper, "parent_id", expected.parent_id, actual.parent_id);
    is_vec_soft_eq(
        helper, "pre_energy", expected.pre_energy, actual.pre_energy);
    is_vec_soft_eq(helper, "pre_time", expected.pre_time, actual.pre_time);
    is_vec_soft_eq(helper, "pre_pos", expected.pre_pos, actual.pre_pos);
    is_vec_soft_eq(helper, "pre_dir", expected.pre_dir, actual.pre_dir);
    is_vec_eq(helper, "pre_physvol", expected.pre_physvol, actual.pre_physvol);
    is_vec_soft_eq(helper, "post_time", expected.post_time, actual.post_time);
    is_vec_soft_eq(
        helper, "post_energy", expected.post_energy, actual.post_energy);
    is_vec_soft_eq(helper, "post_pos", expected.post_pos, actual.post_pos);
    is_vec_soft_eq(helper, "post_dir", expected.post_dir, actual.post_dir);
    is_vec_eq(
        helper, "post_physvol", expected.post_physvol, actual.post_physvol);
    is_vec_eq(helper, "post_status", expected.post_status, actual.post_status);
}

//---------------------------------------------------------------------------//
SimpleSensitiveDetector::SimpleSensitiveDetector(std::string const& name)
    : G4VSensitiveDetector{name}
{
}

//---------------------------------------------------------------------------//
SimpleSensitiveDetector::SimpleSensitiveDetector(G4LogicalVolume const* lv)
    : G4VSensitiveDetector{lv->GetName()}, lv_{lv}
{
}

//---------------------------------------------------------------------------//
bool SimpleSensitiveDetector::ProcessHits(G4Step* step, G4TouchableHistory*)
{
    CELER_EXPECT(step);

    GeantStepView gsv{*step};
    CELER_ASSERT(gsv.has_step_point(StepPoint::pre));
    auto pre_step = gsv.pre_step();

    hits_.step_length.push_back(
        value_as<GeantStepView::Length>(gsv.step_length()) / CLHEP::cm);

    double edep = gsv.energy_deposition().value();

    if (auto* track = step->GetTrack())
    {
        GeantTrackView gtv{*track};
        hits_.particle.push_back(gtv.particle().name());
        double weight = gtv.weight();
        CELER_ASSERT(weight > 0);
        hits_.weight.push_back(weight);
        hits_.track_id.push_back(gtv.track().GetTrackID());
        hits_.parent_id.push_back(gtv.track().GetParentID());
        edep *= weight;
    }
    hits_.energy_deposition.push_back(edep);
    hits_.pre_energy.push_back(value_as<units::ClhepEnergy>(pre_step.energy()));
    hits_.pre_time.push_back(value_as<units::ClhepTime>(pre_step.time()));

    hits_.pre_pos.push_back(native_value_from(pre_step.pos()));
    hits_.pre_dir.push_back(pre_step.dir());

    if (auto* touchable = pre_step.step_point().GetTouchable())
    {
        auto* vol = touchable->GetVolume();
        hits_.pre_physvol.push_back(vol ? vol->GetName() : "<nullptr>");
    }
    if (gsv.has_step_point(StepPoint::post))
    {
        auto post_step = gsv.post_step();
        hits_.post_time.push_back(value_as<units::ClhepTime>(post_step.time()));
        hits_.post_energy.push_back(
            value_as<units::ClhepEnergy>(post_step.energy()));
        hits_.post_pos.push_back(native_value_from(post_step.pos()));
        hits_.post_dir.push_back(post_step.dir());
        if (auto* touchable = post_step.step_point().GetTouchable())
        {
            auto* vol = touchable->GetVolume();
            hits_.post_physvol.push_back(vol ? vol->GetName() : "<nullptr>");
        }
        hits_.post_status.push_back(
            [status = post_step.step_point().GetStepStatus()] {
                switch (status)
                {
                    case G4StepStatus::fWorldBoundary:
                        return "world";
                    case G4StepStatus::fGeomBoundary:
                        return "geo";
                    case G4StepStatus::fUserDefinedLimit:
                        return "user";
                    default:
                        break;
                }
                return "error";
            }());
    }
    return true;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
