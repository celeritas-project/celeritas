//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/fastsim-offload.cc
//---------------------------------------------------------------------------//

#include <algorithm>
#include <iterator>
#include <type_traits>
#include <FTFP_BERT.hh>
#include <G4Box.hh>
#include <G4Electron.hh>
#include <G4FastSimulationPhysics.hh>
#include <G4Gamma.hh>
#include <G4LogicalVolume.hh>
#include <G4Material.hh>
#include <G4PVPlacement.hh>
#include <G4ParticleDefinition.hh>
#include <G4ParticleGun.hh>
#include <G4ParticleTable.hh>
#include <G4Positron.hh>
#include <G4Region.hh>
#include <G4RegionStore.hh>
#include <G4SystemOfUnits.hh>
#include <G4Threading.hh>
#include <G4ThreeVector.hh>
#include <G4Track.hh>
#include <G4TrackStatus.hh>
#include <G4Types.hh>
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
#include <accel/FastSimulationIntegration.hh>
#include <accel/FastSimulationModel.hh>
#include <accel/SetupOptions.hh>
#include <corecel/Macros.hh>
#include <corecel/io/Logger.hh>

using celeritas::FastSimulationIntegration;

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
        CELER_LOG_LOCAL(status) << "Setting up geometry";
        auto* box = new G4Box("world", 100 * cm, 100 * cm, 100 * cm);
        auto* lv = new G4LogicalVolume(box, aluminum_, "world");
        auto* pv = new G4PVPlacement(
            0, G4ThreeVector{}, lv, "world", nullptr, false, 0);
        return pv;
    }

    void ConstructSDandField() final
    {
        CELER_LOG_LOCAL(status)
            << R"(Creating FastSimulationModel for default region)";
        G4Region* default_region = G4RegionStore::GetInstance()->GetRegion(
            "DefaultRegionForTheWorld");
        // Underlying GVFastSimulationModel constructor handles ownership, so
        // we must ignore the returned pointer...
        new celeritas::FastSimulationModel(default_region);
    }

  private:
    G4Material* aluminum_;
};

//---------------------------------------------------------------------------//
// Generate 200 MeV pi+
class PrimaryGeneratorAction final : public G4VUserPrimaryGeneratorAction
{
  public:
    PrimaryGeneratorAction()
    {
        auto g4particle_def
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
        FastSimulationIntegration::Instance().BeginOfRunAction(run);
    }
    void EndOfRunAction(G4Run const* run) final
    {
        FastSimulationIntegration::Instance().EndOfRunAction(run);
    }
};

//---------------------------------------------------------------------------//
class ActionInitialization final : public G4VUserActionInitialization
{
  public:
    void BuildForMaster() const final
    {
        FastSimulationIntegration::Instance().BuildForMaster();

        CELER_LOG_LOCAL(status) << "Constructing user actions";

        this->SetUserAction(new RunAction{});
    }
    void Build() const final
    {
        FastSimulationIntegration::Instance().Build();

        CELER_LOG_LOCAL(status) << "Constructing user actions";

        this->SetUserAction(new PrimaryGeneratorAction{});
        this->SetUserAction(new RunAction{});
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

    // NOTE: since no SD is enabled, we must manually disable Celeritas hit
    // processing
    opts.sd.enabled = false;

    // Use a uniform (zero) magnetic field
    opts.make_along_step = celeritas::UniformAlongStepFactory();

    // Export a GDML file with the problem setup and SDs
    opts.geometry_output_file = "fastsim-offload.gdml";
    // Save diagnostic file to a unique name
    opts.output_file = "fastsim-offload.out.json";
    return opts;
}

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

    // We must add support for fast simulation models to the Physics List
    // NOTE: we have to explicitly name the particles and this should be a
    // superset of what Celeritas can offload
    auto physics_list = new FTFP_BERT{/* verbosity = */ 0};
    auto fast_physics = new G4FastSimulationPhysics();
    fast_physics->ActivateFastSimulation("e-");
    fast_physics->ActivateFastSimulation("e+");
    fast_physics->ActivateFastSimulation("gamma");
    physics_list->RegisterPhysics(fast_physics);
    run_manager->SetUserInitialization(physics_list);

    run_manager->SetUserInitialization(new ActionInitialization());

    FastSimulationIntegration::Instance().SetOptions(MakeOptions());

    run_manager->Initialize();
    run_manager->BeamOn(2);

    return 0;
}
