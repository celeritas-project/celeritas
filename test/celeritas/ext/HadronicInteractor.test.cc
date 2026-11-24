//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/HadronicInteractor.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/ext/HadronicInteractor.hh"

#include <G4DynamicParticle.hh>
#include <G4Electron.hh>
#include <G4ForceCondition.hh>
#include <G4HadFinalState.hh>
#include <G4HadProjectile.hh>
#include <G4HadronicInteraction.hh>
#include <G4HadronicProcess.hh>
#include <G4HadronicProcessType.hh>
#include <G4Material.hh>
#include <G4Nucleus.hh>
#include <G4Step.hh>
#include <G4StepPoint.hh>
#include <G4SystemOfUnits.hh>
#include <G4ThreeVector.hh>
#include <G4Track.hh>
#include <G4VCrossSectionDataSet.hh>
#include <G4VParticleChange.hh>

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// Cross section proportional to energy
class MockXs final : public G4VCrossSectionDataSet
{
  public:
    MockXs() : G4VCrossSectionDataSet("mock-xs") {}

    G4bool IsElementApplicable(
        G4DynamicParticle const*, G4int, G4Material const*) final
    {
        return true;
    }

    G4double GetElementCrossSection(
        G4DynamicParticle const* particle, G4int, G4Material const*) final
    {
        return particle->GetKineticEnergy() / GeV * barn;
    }
};

// Count interactions without changing the projectile
class MockInteraction final : public G4HadronicInteraction
{
  public:
    MockInteraction() : G4HadronicInteraction("mock-interaction") {}

    G4HadFinalState* ApplyYourself(G4HadProjectile const& projectile,
                                   G4Nucleus&) final
    {
        ++count_;

        theParticleChange.Clear();
        theParticleChange.SetStatusChange(isAlive);
        theParticleChange.SetEnergyChange(projectile.GetKineticEnergy());
        theParticleChange.SetMomentumChange(
            projectile.Get4Momentum().vect().unit());
        return &theParticleChange;
    }

    int count() const { return count_; }

  private:
    int count_{0};
};

// Electron that is stepping inside a material
struct MockElectron
{
    G4Track track;
    G4Step step;

    MockElectron(double energy, G4Material* material)
        : track{new G4DynamicParticle(
                    G4Electron::Definition(), G4ThreeVector(0, 0, 1), energy),
                0.0,
                G4ThreeVector(0, 0, 0)}
    {
        step.NewSecondaryVector();
        step.SetTrack(&track);
        step.GetPreStepPoint()->SetMaterial(material);
        track.SetStep(&step);
        track.SetTrackStatus(fAlive);
    }
};

//---------------------------------------------------------------------------//
// An interaction must not be rejected based on the cross section of another
// track. Depending on the version, Geant4 may only apply the integral method
// to hadrons and never reject these interactions.
TEST(HadronicInteractorTest, no_rejection)
{
    // Geant4 registries own the process, cross section, and interaction
    auto* interaction = new MockInteraction;
    auto* process
        = new G4HadronicProcess("mock-electro-nuclear", fHadronInelastic);
    process->AddDataSet(new MockXs);
    process->RegisterMe(interaction);
    // Make the process available for lookup
    process->PreparePhysicsTable(*G4Electron::Definition());
    process->BuildPhysicsTable(*G4Electron::Definition());

    auto* argon = new G4Material(
        "mock-argon", 18., 39.95 * g / mole, 1.4 * g / cm3, kStateLiquid);

    HadronicInteractor interact(*G4Electron::Definition(), fHadronInelastic);

    constexpr int num_samples = 100;
    for (int i = 0; i < num_samples; ++i)
    {
        {
            // Geant4 takes a step with a much higher cross section
            MockElectron other(100 * GeV, argon);
            process->StartTracking(&other.track);
            G4ForceCondition condition;
            process->PostStepGetPhysicalInteractionLength(
                other.track, 0.0, &condition);
        }

        int const prev_count = interaction->count();
        MockElectron electron(1 * GeV, argon);
        G4VParticleChange& result = interact(electron.track);
        EXPECT_EQ(0, result.GetNumberOfSecondaries());
        result.Clear();
        ASSERT_EQ(prev_count + 1, interaction->count()) << "sample " << i;
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
