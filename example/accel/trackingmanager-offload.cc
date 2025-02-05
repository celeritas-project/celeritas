//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file example/accel/trackingmanager-offload.cc
//---------------------------------------------------------------------------//

#include <algorithm>
#include <iterator>
#include <type_traits>
#include <FTFP_BERT.hh>
#include <G4Box.hh>
#include <G4LogicalVolume.hh>
#include <G4Material.hh>
#include <G4PVPlacement.hh>
#include <G4ParticleGun.hh>
#include <G4ParticleTable.hh>
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
#if G4VERSION_NUMBER >= 1100
#    include <G4RunManagerFactory.hh>
#else
#    include <G4MTRunManager.hh>
#endif

#include <accel/AlongStepFactory.hh>
#include <accel/LocalTransporter.hh>
#include <accel/SetupOptions.hh>
#include <accel/SharedParams.hh>
#include <accel/SimpleOffload.hh>
#include <accel/TrackingManagerConstructor.hh>
#include <corecel/Assert.hh>
#include <corecel/Macros.hh>
#include <corecel/io/Logger.hh>

namespace
{
//---------------------------------------------------------------------------//
// Global shared setup options
celeritas::SetupOptions setup_options;
// Shared data and GPU setup
celeritas::SharedParams shared_params;
// Thread-local transporter
G4ThreadLocal celeritas::LocalTransporter local_transporter;

// Simple interface to running celeritas
G4ThreadLocal celeritas::SimpleOffload simple_offload;

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
        setup_options.make_along_step = celeritas::UniformAlongStepFactory();

        // Export a GDML file with the problem setup and SDs
        setup_options.geometry_output_file = "simple-example.gdml";
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
// Generate 100 MeV neutrons
class PrimaryGeneratorAction final : public G4VUserPrimaryGeneratorAction
{
  public:
    PrimaryGeneratorAction()
    {
        auto* g4particle_def
            = G4ParticleTable::GetParticleTable()->FindParticle(2112);
        gun_.SetParticleDefinition(g4particle_def);
        gun_.SetParticleEnergy(100 * MeV);
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
        simple_offload.BeginOfRunAction(run);
    }
    void EndOfRunAction(G4Run const* run) final
    {
        simple_offload.EndOfRunAction(run);
    }
};

//---------------------------------------------------------------------------//
class EventAction final : public G4UserEventAction
{
  public:
    void BeginOfEventAction(G4Event const* event) final
    {
        simple_offload.BeginOfEventAction(event);
    }
    void EndOfEventAction(G4Event const* event) final
    {
        // Log total energy deposition
        if (global_sd)
        {
            CELER_LOG(info) << "Total energy deposited: "
                            << (global_sd->edep() / CLHEP::MeV) << " MeV";
        }
        else
        {
            CELER_LOG(error) << "Global SD was not set";
        }
    }
};

//---------------------------------------------------------------------------//
class ActionInitialization final : public G4VUserActionInitialization
{
  public:
    void BuildForMaster() const final
    {
        simple_offload.BuildForMaster(&setup_options, &shared_params);

        CELER_LOG_LOCAL(status) << "Constructing user actions";

        this->SetUserAction(new RunAction{});
    }
    void Build() const final
    {
        simple_offload.Build(
            &setup_options, &shared_params, &local_transporter);

        CELER_LOG_LOCAL(status) << "Constructing user actions";

        this->SetUserAction(new PrimaryGeneratorAction{});
        this->SetUserAction(new RunAction{});
        this->SetUserAction(new EventAction{});
    }
};

//---------------------------------------------------------------------------//
}  // namespace

int main()
{
    auto run_manager = [] {
#if G4VERSION_NUMBER >= 1100
        return std::unique_ptr<G4RunManager>{
            G4RunManagerFactory::CreateRunManager()};
#else
        return std::make_unique<G4RunManager>();
#endif
    }();

    run_manager->SetUserInitialization(new DetectorConstruction{});

    // Use FTFP_BERT, but use Celeritas tracking for e-/e+/g
    auto* physics_list = new FTFP_BERT{/* verbosity = */ 0};
    physics_list->RegisterPhysics(new celeritas::TrackingManagerConstructor(
        &shared_params, [](int) { return &local_transporter; }));
    run_manager->SetUserInitialization(physics_list);
    run_manager->SetUserInitialization(new ActionInitialization());

    // NOTE: these numbers are appropriate for CPU execution
    setup_options.max_num_tracks = 1024;
    setup_options.initializer_capacity = 1024 * 128;
    // This parameter will eventually be removed
    setup_options.max_num_events = 1024;
    // Celeritas does not support EmStandard MSC physics above 100 MeV
    setup_options.ignore_processes = {"CoulombScat"};
    if (G4VERSION_NUMBER >= 1110)
    {
        // Default Rayleigh scattering 'MinKinEnergyPrim' is no longer
        // consistent
        setup_options.ignore_processes.push_back("Rayl");
    }

    setup_options.output_file = "trackingmanager-offload.out.json";

    run_manager->Initialize();
    run_manager->BeamOn(2);

    return 0;
}
