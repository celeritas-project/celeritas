//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file scripts/ci/test-installation/example-accel.cc
//---------------------------------------------------------------------------//

#include <algorithm>
#include <iterator>
#include <type_traits>
#include <FTFP_BERT.hh>
#include <G4Box.hh>
#include <G4Electron.hh>
#include <G4Gamma.hh>
#include <G4LogicalVolume.hh>
#include <G4Material.hh>
#include <G4PVPlacement.hh>
#include <G4ParticleDefinition.hh>
#include <G4ParticleGun.hh>
#include <G4ParticleTable.hh>
#include <G4Positron.hh>
#include <G4RunManagerFactory.hh>
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
#include <accel/AlongStepFactory.hh>
#include <accel/ExceptionConverter.hh>
#include <accel/LocalTransporter.hh>
#include <accel/Logger.hh>
#include <accel/SetupOptions.hh>
#include <accel/SharedParams.hh>
#include <celeritas/em/msc/UrbanMscParams.hh>
#include <celeritas/global/alongstep/AlongStepGeneralLinearAction.hh>
#include <celeritas/io/ImportData.hh>
#include <corecel/Macros.hh>
#include <corecel/io/Logger.hh>

using celeritas::ExceptionConverter;

namespace
{
//---------------------------------------------------------------------------//
// Global shared setup options
celeritas::SetupOptions setup_options;
// Shared data and GPU setup
celeritas::SharedParams shared_params;
// Thread-local transporter
G4ThreadLocal celeritas::LocalTransporter local_transporter;

//---------------------------------------------------------------------------//
std::shared_ptr<celeritas::ExplicitActionInterface const>
make_nofield_along_step(celeritas::AlongStepFactoryInput const& input)
{
    return celeritas::AlongStepGeneralLinearAction::from_params(
        input.action_id,
        *input.material,
        *input.particle,
        celeritas::UrbanMscParams::from_import(
            *input.particle, *input.material, *input.imported),
        input.imported->em_params.energy_loss_fluct);
}

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
        auto* pv = new G4PVPlacement(
            0, G4ThreeVector{}, lv, "world", nullptr, false, 0);
        return pv;
    }

    void ConstructSDandField() final {}

  private:
    G4Material* aluminum_;
};

//---------------------------------------------------------------------------//
class PrimaryGeneratorAction final : public G4VUserPrimaryGeneratorAction
{
  public:
    PrimaryGeneratorAction()
    {
        auto g4particle_def
            = G4ParticleTable::GetParticleTable()->FindParticle(2112);
        gun_.SetParticleDefinition(g4particle_def);
        gun_.SetParticleEnergy(100 * GeV);
        gun_.SetParticlePosition(G4ThreeVector{0, 0, 0});  // origin
        gun_.SetParticleMomentumDirection(G4ThreeVector{1, 0, 0});  // +x
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

//---------------------------------------------------------------------------//
class RunAction final : public G4UserRunAction
{
  public:
    void BeginOfRunAction(G4Run const* run) final
    {
        ExceptionConverter call_g4exception{"celer0001"};

        if (G4Threading::IsMasterThread())
        {
            CELER_LOG_LOCAL(status) << "Setting up field propagation";
            setup_options.make_along_step = make_nofield_along_step;

            CELER_TRY_HANDLE(shared_params.Initialize(setup_options),
                             call_g4exception);
        }
        else
        {
            CELER_TRY_HANDLE(
                celeritas::SharedParams::InitializeWorker(setup_options),
                call_g4exception);
        }

        if (G4Threading::IsWorkerThread()
            || !G4Threading::IsMultithreadedApplication())
        {
            CELER_TRY_HANDLE(
                local_transporter.Initialize(setup_options, shared_params),
                call_g4exception);
        }
    }
    void EndOfRunAction(G4Run const* run) final
    {
        CELER_LOG_LOCAL(status) << "Finalizing Celeritas";
        ExceptionConverter call_g4exception{"celer0005"};

        if (local_transporter)
        {
            CELER_TRY_HANDLE(local_transporter.Finalize(), call_g4exception);
        }

        if (G4Threading::IsMasterThread())
        {
            CELER_TRY_HANDLE(shared_params.Finalize(), call_g4exception);
        }
    }
};

//---------------------------------------------------------------------------//
class EventAction final : public G4UserEventAction
{
  public:
    void BeginOfEventAction(G4Event const* event) final
    {
        // Set event ID in local transporter
        ExceptionConverter call_g4exception{"celer0002"};
        CELER_TRY_HANDLE(local_transporter.SetEventId(event->GetEventID()),
                         call_g4exception);
    }

    void EndOfEventAction(G4Event const* event) final
    {
        ExceptionConverter call_g4exception{"celer0004"};
        CELER_TRY_HANDLE(local_transporter.Flush(), call_g4exception);
    }
};

//---------------------------------------------------------------------------//
class TrackingAction final : public G4UserTrackingAction
{
    void PreUserTrackingAction(G4Track const* track) final
    {
        static G4ParticleDefinition const* const allowed_particles[] = {
            G4Gamma::Gamma(),
            G4Electron::Electron(),
            G4Positron::Positron(),
        };

        if (std::find(std::begin(allowed_particles),
                      std::end(allowed_particles),
                      track->GetDefinition())
            != std::end(allowed_particles))
        {
            // Celeritas is transporting this track
            ExceptionConverter call_g4exception{"celer0003"};
            CELER_TRY_HANDLE(local_transporter.Push(*track), call_g4exception);
            const_cast<G4Track*>(track)->SetTrackStatus(fStopAndKill);
        }
    }
};

//---------------------------------------------------------------------------//
class ActionInitialization final : public G4VUserActionInitialization
{
  public:
    void BuildForMaster() const final { this->Build(); }
    void Build() const final
    {
        CELER_LOG_LOCAL(status) << "Constructing user actions";

        this->SetUserAction(new PrimaryGeneratorAction{});
        this->SetUserAction(new RunAction{});
        this->SetUserAction(new EventAction{});
        this->SetUserAction(new TrackingAction{});
    }
};

//---------------------------------------------------------------------------//
}  // namespace

int main()
{
    std::unique_ptr<G4RunManager> run_manager{
        G4RunManagerFactory::CreateRunManager(G4RunManagerType::SerialOnly)};

    celeritas::self_logger() = celeritas::make_mt_logger(*run_manager);
    run_manager->SetUserInitialization(new DetectorConstruction{});
    run_manager->SetUserInitialization(new FTFP_BERT{/* verbosity = */ 0});
    run_manager->SetUserInitialization(new ActionInitialization());

    setup_options.max_num_tracks = 1024;
    setup_options.max_num_events = 1024;
    setup_options.initializer_capacity = 1024 * 128;
    setup_options.secondary_stack_factor = 3.0;
    setup_options.ignore_processes
        = {"CoulombScat", "muIoni", "muBrems", "muPairProd"};

    run_manager->Initialize();
    run_manager->BeamOn(1);

    return 0;
}
