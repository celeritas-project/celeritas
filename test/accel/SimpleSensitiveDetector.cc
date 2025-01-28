//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/SimpleSensitiveDetector.cc
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
#include "corecel/io/Repr.hh"

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

            "static double const expected_pre_energy[] = "
         << repr(this->pre_energy)
         << ";\n"
            "EXPECT_VEC_SOFT_EQ(expected_pre_energy, result.pre_energy);\n"

            "static double const expected_pre_pos[] = "
         << repr(this->pre_pos)
         << ";\n"
            "EXPECT_VEC_SOFT_EQ(expected_pre_pos, result.pre_pos);\n"

            "static char const* const expected_pre_physvol[] = "
         << repr(this->pre_physvol)
         << ";\n"
            "EXPECT_VEC_EQ(expected_pre_physvol, result.pre_physvol);\n"

            "static double const expected_post_time[] = "
         << repr(this->post_time)
         << ";\n"
            "EXPECT_VEC_SOFT_EQ(expected_post_time, result.post_time);\n"
            "/*** END CODE ***/\n";
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

    hits_.energy_deposition.push_back(step->GetTotalEnergyDeposit()
                                      / CLHEP::MeV);
    hits_.step_length.push_back(step->GetStepLength() / CLHEP::cm);
    if (auto* track = step->GetTrack())
    {
        hits_.particle.push_back(track->GetDefinition()->GetParticleName());
    }

    hits_.pre_energy.push_back(pre_step->GetKineticEnergy() / CLHEP::MeV);

    for (int i : range(3))
    {
        hits_.pre_pos.push_back(pre_step->GetPosition()[i] / CLHEP::cm);
    }

    if (auto* touchable = pre_step->GetTouchable())
    {
        auto* vol = touchable->GetVolume();
        hits_.pre_physvol.push_back(vol ? vol->GetName() : "<nullptr>");
    }
    hits_.post_time.push_back(step->GetPostStepPoint()->GetGlobalTime()
                              / CLHEP::ns);
    return true;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
