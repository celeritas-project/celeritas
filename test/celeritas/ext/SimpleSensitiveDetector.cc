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
#include "corecel/math/Quantity.hh"
#include "corecel/math/SoftEqual.hh"
#include "geocel/UnitUtils.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/ext/GeantStepPointView.hh"
#include "celeritas/ext/GeantStepView.hh"
#include "celeritas/ext/GeantTrackView.hh"

#include "Comparer.hh"
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
bool check_ref_eq(AssertionHelper& helper,
                  SimpleHitsResult const& expected,
                  SimpleHitsResult const& actual)
{
    RefComparer<SimpleHitsResult> compare{helper, expected, actual};
    constexpr auto tol
        = static_cast<double>(SoftEqual<celeritas::real_type>{}.rel());
    CELER_INVOKE_ATTR(
        compare.is_soft_eq, SimpleHitsResult, energy_deposition, tol);
    CELER_INVOKE_ATTR(compare.is_soft_eq, SimpleHitsResult, step_length, tol);
    CELER_INVOKE_ATTR(compare.is_eq, SimpleHitsResult, particle);
    CELER_INVOKE_ATTR(compare.is_soft_eq, SimpleHitsResult, weight, tol);
    CELER_INVOKE_ATTR(compare.is_eq, SimpleHitsResult, track_id);
    CELER_INVOKE_ATTR(compare.is_eq, SimpleHitsResult, parent_id);
    CELER_INVOKE_ATTR(compare.is_soft_eq, SimpleHitsResult, pre_energy, tol);
    CELER_INVOKE_ATTR(compare.is_soft_eq, SimpleHitsResult, pre_time, tol);
    CELER_INVOKE_ATTR(compare.is_soft_eq, SimpleHitsResult, pre_pos, tol);
    CELER_INVOKE_ATTR(compare.is_soft_eq, SimpleHitsResult, pre_dir, tol);
    CELER_INVOKE_ATTR(compare.is_eq, SimpleHitsResult, pre_physvol);
    CELER_INVOKE_ATTR(compare.is_soft_eq, SimpleHitsResult, post_time, tol);
    CELER_INVOKE_ATTR(compare.is_soft_eq, SimpleHitsResult, post_energy, tol);
    CELER_INVOKE_ATTR(compare.is_soft_eq, SimpleHitsResult, post_pos, tol);
    CELER_INVOKE_ATTR(compare.is_soft_eq, SimpleHitsResult, post_dir, tol);
    CELER_INVOKE_ATTR(compare.is_eq, SimpleHitsResult, post_physvol);
    CELER_INVOKE_ATTR(compare.is_eq, SimpleHitsResult, post_status);
    return static_cast<bool>(compare);
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

    auto append_pos = [](auto& hits, GeantStepPointView::Length3 g4pos) {
        hits.push_back(to_cm(g4pos));
    };
    auto append_dir = [](auto& hits, GeantStepPointView::Real3 g4dir) {
        hits.push_back(g4dir);
    };

    auto append_touchable = [](auto& hits, auto* touchable) {
        if (touchable)
        {
            auto* vol = touchable->GetVolume();
            hits.push_back(vol ? vol->GetName() : "<nullptr>");
        }
        else
        {
            hits.push_back("<error>");
        }
    };
    GeantStepView gsv{*step};

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

    if (gsv.has_step_point(StepPoint::pre))
    {
        auto step = gsv.pre_step();
        hits_.pre_energy.push_back(value_as<units::ClhepEnergy>(step.energy()));
        hits_.pre_time.push_back(value_as<units::ClhepTime>(step.time()));
        append_pos(hits_.pre_pos, step.pos());
        append_dir(hits_.pre_dir, step.dir());
        append_touchable(hits_.pre_physvol, step.step_point().GetTouchable());
    }
    if (gsv.has_step_point(StepPoint::post))
    {
        auto step = gsv.post_step();
        hits_.post_time.push_back(value_as<units::ClhepTime>(step.time()));
        hits_.post_energy.push_back(
            value_as<units::ClhepEnergy>(step.energy()));
        append_pos(hits_.post_pos, step.pos());
        append_dir(hits_.post_dir, step.dir());
        append_touchable(hits_.post_physvol, step.step_point().GetTouchable());
        hits_.post_status.push_back(
            [status = step.step_point().GetStepStatus()] {
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
