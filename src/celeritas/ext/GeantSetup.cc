//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantSetup.cc
//---------------------------------------------------------------------------//
#include "GeantSetup.hh"

#include <memory>
#include <utility>
#include <G4Version.hh>
#if G4VERSION_NUMBER >= 1070
#    include <G4HadronicParameters.hh>
#else
#    include <G4HadronicProcessStore.hh>
#endif
#include <G4ParticleTable.hh>
#include <G4RunManager.hh>
#include <G4VPhysicalVolume.hh>
#include <G4VSensitiveDetector.hh>
#include <G4VUserDetectorConstruction.hh>
#if G4VERSION_NUMBER >= 1100
#    include <G4RunManagerFactory.hh>
#endif

#include "corecel/io/Logger.hh"
#include "corecel/io/ScopedTimeAndRedirect.hh"
#include "corecel/io/ScopedTimeLog.hh"
#include "corecel/sys/ScopedMem.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "geocel/GeantGeoParams.hh"
#include "geocel/GeantUtils.hh"
#include "geocel/ScopedGeantExceptionHandler.hh"
#include "geocel/ScopedGeantLogger.hh"
#include "celeritas/g4/DetectorConstruction.hh"

#include "EmPhysicsList.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
//! Placeholder SD class for generating model data from GDML
class GdmlSensitiveDetector final : public G4VSensitiveDetector
{
  public:
    GdmlSensitiveDetector(std::string const& name) : G4VSensitiveDetector{name}
    {
    }

    void Initialize(G4HCofThisEvent*) final {}
    bool ProcessHits(G4Step*, G4TouchableHistory*) final { return false; }
};

}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct from a GDML file and physics options.
 */
GeantSetup::GeantSetup(std::string const& gdml_filename, Options options)
    : GeantSetup{gdml_filename, std::move(options), {}}
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct from a GDML file and physics options.
 */
GeantSetup::GeantSetup(std::string const& gdml_filename,
                       Options options,
                       SetString sd_names)
{
    CELER_LOG(status) << "Initializing Geant4 run manager";
    ScopedProfiling profile_this{"initialize-geant"};
    ScopedMem record_setup_mem("GeantSetup.construct");

    {
        // Run manager writes output that cannot be redirected with
        // GeantLoggerAdapter: capture all output from this section
        ScopedTimeAndRedirect scoped_time{"G4RunManager"};
        ScopedGeantExceptionHandler scoped_exceptions;

        // Access the particle table before creating the run manager, so that
        // missing environment variables like G4ENSDFSTATEDATA get caught
        // cleanly rather than segfaulting
        G4ParticleTable::GetParticleTable();

        // Guard against segfaults due to bad Geant4 global cleanup
        static int geant_launch_count = 0;
        CELER_VALIDATE(geant_launch_count == 0,
                       << "Geant4 cannot be 'run' more than once per "
                          "execution");
        ++geant_launch_count;

#if G4VERSION_NUMBER >= 1100
        run_manager_.reset(
            G4RunManagerFactory::CreateRunManager(G4RunManagerType::Serial));
#else
        // Note: custom deleter means `make_unique` won't work
        run_manager_.reset(new G4RunManager);
#endif
        CELER_ASSERT(run_manager_);
    }

    ScopedGeantLogger scoped_logger(celeritas::world_logger());
    ScopedGeantExceptionHandler scoped_exceptions;

    {
        CELER_LOG(status) << "Initializing Geant4 geometry and physics list";

        // Construct the geometry
        DetectorConstruction::SDBuilder make_sd;
        if (!sd_names.empty())
        {
            using UPSD = DetectorConstruction::UPSD;
            make_sd = [all_sd = std::move(sd_names)](
                          std::string const& name) -> UPSD {
                if (!all_sd.count(name))
                    return nullptr;
                // Create an SD for the corresponding SensDet aux value
                return std::make_unique<GdmlSensitiveDetector>(name);
            };
        }
        auto detector = std::make_unique<DetectorConstruction>(
            gdml_filename, std::move(make_sd));
        run_manager_->SetUserInitialization(detector.release());

        // Construct the physics
        auto physics_list = std::make_unique<EmPhysicsList>(options);
        run_manager_->SetUserInitialization(physics_list.release());
    }

    {
        CELER_LOG(status) << "Building Geant4 physics tables";
        ScopedMem record_mem("GeantSetup.initialize");
        ScopedTimeLog scoped_time;

        // Suppress Geant4 verbosity when G4ProcessType::fHadronic are enabled
#if G4VERSION_NUMBER >= 1070
        auto* had_params = G4HadronicParameters::Instance();
        had_params->SetVerboseLevel(0);
#else
        auto had_store = G4HadronicProcessStore::Instance();
        had_store->SetVerbose(0);
#endif

        run_manager_->Initialize();
        run_manager_->RunInitialization();
    }

    {
        // Save the geometry
        auto* detcon = dynamic_cast<DetectorConstruction const*>(
            run_manager_->GetUserDetectorConstruction());
        CELER_ASSERT(detcon);
        this->geo_ = detcon->geo();
    }

    CELER_ENSURE(this->geo_);
    CELER_ENSURE(*this);
}

//---------------------------------------------------------------------------//
/*!
 * Terminate the run manager on destruction.
 */
GeantSetup::~GeantSetup()
{
    if (run_manager_)
    {
        run_manager_->RunTermination();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Delete a geant4 run manager.
 */
void GeantSetup::RMDeleter::operator()(G4RunManager* rm) const
{
    CELER_LOG(debug) << "Clearing Geant4 state";
    delete rm;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
