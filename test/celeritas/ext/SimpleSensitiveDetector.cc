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
#include <G4ParticleDefinition.hh>
#include <G4Step.hh>
#include <G4TouchableHistory.hh>
#include <G4Track.hh>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/Repr.hh"
#include "celeritas/ext/GeantParticleView.hh"

using std::cout;

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
void SimpleHitsResult::print_expected() const
{
    cout << "/*** ADD THE FOLLOWING UNIT TEST CODE ***/\n"
            "static double const expected_energy_deposition[] = "
         << repr(this->energy_deposition)
         << ";\n"
            "EXPECT_VEC_SOFT_EQ(expected_energy_deposition, "
            "result.energy_deposition);\n"

            "static double const expected_step_length[] = "
         << repr(this->step_length)
         << ";\n"
            "EXPECT_VEC_SOFT_EQ(expected_step_length, "
            "result.step_length);\n"

            "static char const* const expected_particle[] = "
         << repr(this->particle)
         << ";\n"
            "EXPECT_VEC_EQ(expected_particle, result.particle);\n"

            "static double const expected_weight[] = "
         << repr(this->weight)
         << ";\n"
            "EXPECT_VEC_SOFT_EQ(expected_weight, result.weight);\n"

            "static int const expected_track_id[] = "
         << repr(this->track_id)
         << ";\n"
            "EXPECT_VEC_EQ(expected_track_id, result.track_id);\n"

            "static int const expected_parent_id[] = "
         << repr(this->parent_id)
         << ";\n"
            "EXPECT_VEC_EQ(expected_parent_id, result.parent_id);\n"

            "static double const expected_pre_energy[] = "
         << repr(this->pre_energy)
         << ";\n"
            "EXPECT_VEC_SOFT_EQ(expected_pre_energy, result.pre_energy);\n"

            "static double const expected_pre_time[] = "
         << repr(this->pre_time)
         << ";\n"
            "EXPECT_VEC_SOFT_EQ(expected_pre_time, result.pre_time);\n"

            "static double const expected_pre_pos[] = "
         << repr(this->pre_pos)
         << ";\n"
            "EXPECT_VEC_SOFT_EQ(expected_pre_pos, result.pre_pos);\n"

            "static double const expected_pre_dir[] = "
         << repr(this->pre_dir)
         << ";\n"
            "EXPECT_VEC_SOFT_EQ(expected_pre_dir, result.pre_dir);\n"

            "static char const* const expected_pre_physvol[] = "
         << repr(this->pre_physvol)
         << ";\n"
            "EXPECT_VEC_EQ(expected_pre_physvol, result.pre_physvol);\n"

            "static double const expected_post_time[] = "
         << repr(this->post_time)
         << ";\n"
            "EXPECT_VEC_SOFT_EQ(expected_post_time, result.post_time);\n"

            "static double const expected_post_energy[] = "
         << repr(this->post_energy)
         << ";\n"
            "EXPECT_VEC_SOFT_EQ(expected_post_energy, result.post_energy);\n"

            "static double const expected_post_pos[] = "
         << repr(this->post_pos)
         << ";\n"
            "EXPECT_VEC_SOFT_EQ(expected_post_pos, result.post_pos);\n"

            "static double const expected_post_dir[] = "
         << repr(this->post_dir)
         << ";\n"
            "EXPECT_VEC_SOFT_EQ(expected_post_dir, result.post_dir);\n"

            "static char const* const expected_post_physvol[] = "
         << repr(this->post_physvol)
         << ";\n"
            "EXPECT_VEC_EQ(expected_post_physvol, result.post_physvol);\n"

            "static char const* const expected_post_status[] = "
         << repr(this->post_status)
         << ";\n"
            "EXPECT_VEC_EQ(expected_post_status, result.post_status);\n"

            "/*** END CODE ***/\n";
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

    auto* pre_step = step->GetPreStepPoint();
    CELER_ASSERT(pre_step);

    hits_.step_length.push_back(step->GetStepLength() / CLHEP::cm);

    double edep = step->GetTotalEnergyDeposit() / CLHEP::MeV;

    if (auto* track = step->GetTrack())
    {
        GeantParticleView pv{*track->GetParticleDefinition()};
        hits_.particle.push_back(pv.name());
        double weight = track->GetWeight();
        CELER_ASSERT(weight > 0);
        hits_.weight.push_back(weight);
        hits_.track_id.push_back(track->GetTrackID());
        hits_.parent_id.push_back(track->GetParentID());
        edep *= weight;
    }
    hits_.energy_deposition.push_back(edep);
    hits_.pre_energy.push_back(pre_step->GetKineticEnergy() / CLHEP::MeV);
    hits_.pre_time.push_back(pre_step->GetGlobalTime() / CLHEP::ns);

    for (int i : range(3))
    {
        hits_.pre_pos.push_back(pre_step->GetPosition()[i] / CLHEP::cm);
    }
    for (int i : range(3))
    {
        hits_.pre_dir.push_back(pre_step->GetMomentumDirection()[i]);
    }

    if (auto* touchable = pre_step->GetTouchable())
    {
        auto* vol = touchable->GetVolume();
        hits_.pre_physvol.push_back(vol ? vol->GetName() : "<nullptr>");
    }
    hits_.post_time.push_back(step->GetPostStepPoint()->GetGlobalTime()
                              / CLHEP::ns);

    if (auto* post_step = step->GetPostStepPoint())
    {
        hits_.post_energy.push_back(post_step->GetKineticEnergy() / CLHEP::MeV);
        for (int i : range(3))
        {
            hits_.post_pos.push_back(post_step->GetPosition()[i] / CLHEP::cm);
        }
        for (int i : range(3))
        {
            hits_.post_dir.push_back(post_step->GetMomentumDirection()[i]);
        }
        if (auto* touchable = post_step->GetTouchable())
        {
            auto* vol = touchable->GetVolume();
            hits_.post_physvol.push_back(vol ? vol->GetName() : "<nullptr>");
        }
        hits_.post_status.push_back([status = post_step->GetStepStatus()] {
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
