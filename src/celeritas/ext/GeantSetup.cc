//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantSetup.cc
//---------------------------------------------------------------------------//
#include "GeantSetup.hh"

#include <memory>
#include <utility>
#include <G4ParticleTable.hh>
#include <G4RunManager.hh>
#include <G4VPhysicalVolume.hh>
#include <G4VUserDetectorConstruction.hh>
#include <G4Version.hh>
#if G4VERSION_NUMBER >= 1070
#    include <G4Backtrace.hh>
#endif
#if G4VERSION_NUMBER >= 1100
#    include <G4RunManagerFactory.hh>
#endif

#include "corecel/io/Logger.hh"
#include "corecel/io/ScopedTimeAndRedirect.hh"
#include "corecel/io/ScopedTimeLog.hh"
#include "corecel/sys/ScopedMem.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "geocel/GeantGdmlLoader.hh"
#include "geocel/GeantGeoParams.hh"
#include "geocel/GeantUtils.hh"
#include "geocel/ScopedGeantExceptionHandler.hh"
#include "geocel/ScopedGeantLogger.hh"

#include "EmPhysicsList.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Load the detector geometry from a GDML input file.
 */
class DetectorConstruction : public G4VUserDetectorConstruction
{
  public:
    explicit DetectorConstruction(G4VPhysicalVolume* world) : world_{world}
    {
        CELER_ENSURE(world_);
    }

    G4VPhysicalVolume* Construct() override { return world_; }

  private:
    G4VPhysicalVolume* world_;
};

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct from a GDML file and physics options.
 */
GeantSetup::GeantSetup(std::string const& gdml_filename, Options options)
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

        // Disable signal handling
        disable_geant_signal_handler();

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

    G4VPhysicalVolume* world{nullptr};
    {
        CELER_LOG(status) << "Initializing Geant4 geometry and physics list";

        // Load GDML and reference the world pointer
        // TODO: pass GdmlLoader options through SetupOptions
        world = load_gdml(gdml_filename);
        CELER_ASSERT(world);

        // Construct the geometry
        auto detector = std::make_unique<DetectorConstruction>(world);
        run_manager_->SetUserInitialization(detector.release());

        // Construct the physics
        auto physics_list = std::make_unique<EmPhysicsList>(options);
        run_manager_->SetUserInitialization(physics_list.release());
    }

    {
        CELER_LOG(status) << "Building Geant4 physics tables";
        ScopedMem record_mem("GeantSetup.initialize");
        ScopedTimeLog scoped_time;

        run_manager_->Initialize();
        run_manager_->RunInitialization();
    }

    {
        // Create non-owning Geant4 geo wrapper and save as tracking geometry
        geo_ = std::make_shared<GeantGeoParams>(world, Ownership::reference);
        celeritas::geant_geo(geo_);
    }

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
