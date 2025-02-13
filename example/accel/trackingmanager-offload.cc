//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/trackingmanager-offload.cc
//---------------------------------------------------------------------------//

#include <algorithm>
#include <iterator>
#include <type_traits>

// Geant4
#include <FTFP_BERT.hh>
#include <G4Box.hh>
#include <G4LogicalVolume.hh>
#include <G4Material.hh>
#include <G4PVPlacement.hh>
#include <G4ParticleGun.hh>
#include <G4ParticleTable.hh>
#include <G4RunManagerFactory.hh>
#include <G4SDManager.hh>
#include <G4SystemOfUnits.hh>
#include <G4ThreeVector.hh>
#include <G4UserEventAction.hh>
#include <G4UserRunAction.hh>
#include <G4UserTrackingAction.hh>
#include <G4VUserActionInitialization.hh>
#include <G4VUserDetectorConstruction.hh>
#include <G4VUserPrimaryGeneratorAction.hh>
#include <G4Version.hh>

// Celeritas
#include <accel/AlongStepFactory.hh>
#include <accel/LocalTransporter.hh>
#include <accel/SetupOptions.hh>
#include <accel/TrackingManagerConstructor.hh>
#include <accel/TrackingManagerIntegration.hh>
#include <corecel/Assert.hh>
#include <corecel/Macros.hh>
#include <corecel/io/Logger.hh>

namespace
{
//---------------------------------------------------------------------------//
class SensitiveDetector final : public G4VSensitiveDetector
{
  public:
    explicit SensitiveDetector(std::string name)
        : G4VSensitiveDetector{std::move(name)}
    {
    }

    double edep() const { return edep_; }

  protected:
    void Initialize(G4HCofThisEvent*) final { edep_ = 0; }
    bool ProcessHits(G4Step* step, G4TouchableHistory*) final
    {
        CELER_ASSERT(step);
        edep_ += step->GetTotalEnergyDeposit();
        return true;
    }

  private:
    double edep_{0};
};

// Simple (not best practice) way of accessing SD
G4ThreadLocal SensitiveDetector const* global_sd{nullptr};

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
        auto* box = new G4Box("world", 100 * cm, 100 * cm, 100 * cm);
        auto* lv = new G4LogicalVolume(box, aluminum_, "world");
        world_lv_ = lv;
        auto* pv = new G4PVPlacement(
            0, G4ThreeVector{}, lv, "world", nullptr, false, 0);
        return pv;
    }

    void ConstructSDandField() final
    {
        auto* sd_manager = G4SDManager::GetSDMpointer();
        auto detector = std::make_unique<SensitiveDetector>("example-sd");
        world_lv_->SetSensitiveDetector(detector.get());
        global_sd = detector.get();
        sd_manager->AddNewDetector(detector.release());
    }

  private:
    G4Material* aluminum_{nullptr};
    G4LogicalVolume* world_lv_{nullptr};
};

//---------------------------------------------------------------------------//
// Generate 200 MeV pi+
class PrimaryGeneratorAction final : public G4VUserPrimaryGeneratorAction
{
  public:
    PrimaryGeneratorAction()
    {
        auto* g4particle_def
            = G4ParticleTable::GetParticleTable()->FindParticle(211);
        gun_.SetParticleDefinition(g4particle_def);
        gun_.SetParticleEnergy(200 * MeV);
        gun_.SetParticlePosition(G4ThreeVector{0, 0, 0});  // origin
        gun_.SetParticleMomentumDirection(G4ThreeVector{1, 0, 0});  // +x
    }

    void GeneratePrimaries(G4Event* event) final
    {
        CELER_LOG_LOCAL(status) << "Generating primaries";
        gun_.GeneratePrimaryVertex(event);
    }

  private:
    G4ParticleGun gun_;
};

//---------------------------------------------------------------------------//
class RunAction final : public G4UserRunAction
{
  public:
    void BeginOfRunAction(G4Run const* run) final
    {
        celeritas::TrackingManagerIntegration::Instance().BeginOfRunAction(run);
    }
    void EndOfRunAction(G4Run const* run) final
    {
        celeritas::TrackingManagerIntegration::Instance().EndOfRunAction(run);
    }
};

//---------------------------------------------------------------------------//
class EventAction final : public G4UserEventAction
{
  public:
    void BeginOfEventAction(G4Event const*) final {}
    void EndOfEventAction(G4Event const* event) final
    {
        // Log total energy deposition
        if (global_sd)
        {
            CELER_LOG_LOCAL(info)
                << "Total energy deposited for event " << event->GetEventID()
                << ": " << (global_sd->edep() / CLHEP::MeV) << " MeV";
        }
        else
        {
            CELER_LOG_LOCAL(error) << "Global SD was not set";
        }
    }
};

//---------------------------------------------------------------------------//
class ActionInitialization final : public G4VUserActionInitialization
{
  public:
    void BuildForMaster() const final
    {
        celeritas::TrackingManagerIntegration::Instance().BuildForMaster();

        CELER_LOG_LOCAL(status) << "Constructing user actions";

        this->SetUserAction(new RunAction{});
    }
    void Build() const final
    {
        celeritas::TrackingManagerIntegration::Instance().Build();

        CELER_LOG_LOCAL(status) << "Constructing user actions";

        this->SetUserAction(new PrimaryGeneratorAction{});
        this->SetUserAction(new RunAction{});
        this->SetUserAction(new EventAction{});
    }
};

//---------------------------------------------------------------------------//
/*!
 * Construct options for Celeritas.
 */
celeritas::SetupOptions MakeOptions()
{
    celeritas::SetupOptions opts;
    // NOTE: these numbers are appropriate for CPU execution
    opts.max_num_tracks = 2024;
    opts.initializer_capacity = 2024 * 128;
    // Celeritas does not support EmStandard MSC physics above 200 MeV
    opts.ignore_processes = {"CoulombScat"};

    // Use a uniform (zero) magnetic field
    opts.make_along_step = celeritas::UniformAlongStepFactory();

    // Export a GDML file with the problem setup and SDs
    opts.geometry_output_file = "simple-example.gdml";
    // Save diagnostic file to a unique name
    opts.output_file = "trackingmanager-offload.out.json";
    return opts;
}

//---------------------------------------------------------------------------//
}  // namespace

int main()
{
    auto run_manager = std::unique_ptr<G4RunManager>{
        G4RunManagerFactory::CreateRunManager()};

    run_manager->SetUserInitialization(new DetectorConstruction{});

    auto& tmi = celeritas::TrackingManagerIntegration::Instance();

    // Use FTFP_BERT, but use Celeritas tracking for e-/e+/g
    auto* physics_list = new FTFP_BERT{/* verbosity = */ 0};
    physics_list->RegisterPhysics(
        new celeritas::TrackingManagerConstructor(&tmi));
    run_manager->SetUserInitialization(physics_list);
    run_manager->SetUserInitialization(new ActionInitialization());

    tmi.SetOptions(MakeOptions());

    run_manager->Initialize();
    run_manager->BeamOn(2);

    return 0;
}
