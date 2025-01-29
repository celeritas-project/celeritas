//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/IntegrationTestBase.cc
//---------------------------------------------------------------------------//
#include "IntegrationTestBase.hh"

#include <G4Box.hh>
#include <G4PVPlacement.hh>
#include <G4ParticleGun.hh>
#include <G4SDManager.hh>
#include <G4SystemOfUnits.hh>
#include <G4ThreeVector.hh>
#include <G4VUserDetectorConstruction.hh>
#include <G4VUserPrimaryGeneratorAction.hh>
#include <G4Version.hh>
#if G4VERSION_NUMBER >= 1100
#    include <G4RunManagerFactory.hh>
#else
#    include <G4MTRunManager.hh>
#endif

#include "corecel/io/Logger.hh"

#include "SimpleSensitiveDetector.hh"

namespace celeritas
{
namespace test
{
namespace
{
//---------------------------------------------------------------------------//
class DetectorConstruction final : public G4VUserDetectorConstruction
{
  public:
    DetectorConstruction()
        : aluminum_{new G4Material{
              "Aluminium", 13., 26.98 * g / mole, 2.700 * g / cm3}}
    {
    }

    G4VPhysicalVolume* Construct() final
    {
        CELER_LOG_LOCAL(status) << "Setting up detector";
        auto* box = new G4Box("world", 1000 * cm, 1000 * cm, 1000 * cm);
        auto* lv = new G4LogicalVolume(box, aluminum_, "world");
        world_lv_ = lv;
        auto* pv = new G4PVPlacement(
            0, G4ThreeVector{}, lv, "world", nullptr, false, 0);
        return pv;
    }

    void ConstructSDandField() final
    {
        CELER_LOG_LOCAL(status) << "Constructing SD";
        auto* sd_manager = G4SDManager::GetSDMpointer();
        auto detector = std::make_unique<SimpleSensitiveDetector>(world_lv_);
        world_lv_->SetSensitiveDetector(detector.get());
        sd_manager->AddNewDetector(detector.release());
    }

  private:
    G4Material* aluminum_{nullptr};
    G4LogicalVolume* world_lv_{nullptr};
};

//---------------------------------------------------------------------------//
// TODO: make isotropic instead of unidirectional
class PrimaryGeneratorAction final : public G4VUserPrimaryGeneratorAction
{
  public:
    using Energy = units::MevEnergy;

    PrimaryGeneratorAction(PDGNumber pdg, Energy energy)
    {
        auto g4particle_def
            = G4ParticleTable::GetParticleTable()->FindParticle(pdg.get());
        CELER_VALIDATE(g4particle_def,
                       << "particle " << pdg.get() << " not found");
        gun_.SetParticleDefinition(g4particle_def);
        gun_.SetParticleEnergy(energy.value() * MeV);
        gun_.SetParticlePosition(G4ThreeVector{0, 0, 0});  // origin
        gun_.SetParticleMomentumDirection(G4ThreeVector{0, 0, 1});  // +z
    }

    // Generate 100 GeV neutrons
    void GeneratePrimaries(G4Event* event) final
    {
        CELER_LOG_LOCAL(status) << "Generating primaries";
        gun_.GeneratePrimaryVertex(event);
    }

  private:
    G4ParticleGun gun_;
};

}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Create or access the run manager (created once per execution).
 */
G4RunManager& IntegrationTestBase::run_manager()
{
    static std::unique_ptr<G4RunManager> const rm = [] {
#if G4VERSION_NUMBER >= 1100
        return std::unique_ptr<G4RunManager>{
            G4RunManagerFactory::CreateRunManager()};
#else
        return std::make_unique<G4RunManager>();
#endif
    }();

    CELER_ENSURE(rm);
    return *rm;
}

//---------------------------------------------------------------------------//
/*!
 * Create geometry helper.
 */
auto IntegrationTestBase::make_detector_construction() -> UPDetector
{
    return std::make_unique<DetectorConstruction>();
}

//---------------------------------------------------------------------------//
/*!
 * Create primary generator, isotropic at origin.
 */
auto IntegrationTestBase::make_primaries(PDGNumber pdg,
                                         Energy energy) -> UPPrimary
{
    CELER_EXPECT(pdg);
    CELER_EXPECT(energy > zero_quantity());
    return std::make_unique<PrimaryGeneratorAction>(pdg, energy);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
