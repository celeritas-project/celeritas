//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantSetup.cc
//---------------------------------------------------------------------------//
#include "GeantSetup.hh"

#include <memory>
#include <G4Event.hh>
#include <G4ParticleGun.hh>
#include <G4ParticleTable.hh>
#include <G4SystemOfUnits.hh>
#include <G4VUserActionInitialization.hh>
#include <G4VUserDetectorConstruction.hh>
#include <G4VUserPrimaryGeneratorAction.hh>

#include "GeantVersion.hh"
#if CELERITAS_G4_V10
#    include <G4RunManager.hh>
#else
#    include <G4RunManagerFactory.hh>
#endif

#include "corecel/io/ScopedTimeAndRedirect.hh"
#include "corecel/io/ScopedTimeLog.hh"

#include "LoadGdml.hh"
#include "detail/GeantExceptionHandler.hh"
#include "detail/GeantLoggerAdapter.hh"
#include "detail/GeantPhysicsList.hh"

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
    explicit DetectorConstruction(UPG4PhysicalVolume world)
        : world_{std::move(world)}
    {
        CELER_ENSURE(world_);
    }

    G4VPhysicalVolume* Construct() override
    {
        CELER_EXPECT(world_);
        return world_.release();
    }

  private:
    UPG4PhysicalVolume world_;
};

//---------------------------------------------------------------------------//
} // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct from a GDML file and physics options.
 */
GeantSetup::GeantSetup(const std::string& gdml_filename, Options options)
{
    CELER_LOG(status) << "Initializing Geant4 run manager";

    {
        // Run manager writes output that cannot be redirected...
        ScopedTimeAndRedirect         scoped_time("G4RunManager");
        detail::GeantExceptionHandler scoped_exception_handler;
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

#if CELERITAS_G4_V10
        // Note: custom deleter means `make_unique` won't work
        run_manager_.reset(new G4RunManager);
#else
        run_manager_.reset(
            G4RunManagerFactory::CreateRunManager(G4RunManagerType::Serial));
#endif
        CELER_ASSERT(run_manager_);
    }

    detail::GeantLoggerAdapter    scoped_logger;
    detail::GeantExceptionHandler scoped_exception_handler;

    {
        CELER_LOG(status) << "Initializing Geant4 geometry and physics";
        ScopedTimeLog scoped_time;

        // Load GDML and save a copy of the pointer
        auto world = load_gdml(gdml_filename);
        CELER_ASSERT(world);
        world_ = world.get();

        // Construct the geometry
        auto detector
            = std::make_unique<DetectorConstruction>(std::move(world));
        run_manager_->SetUserInitialization(detector.release());

        // Construct the physics
        auto physics_list = std::make_unique<detail::GeantPhysicsList>(options);
        run_manager_->SetUserInitialization(physics_list.release());
    }

    {
        CELER_LOG(status) << "Initializing Geant4 physics tables";
        ScopedTimeLog scoped_time;

        run_manager_->Initialize();
        run_manager_->RunInitialization();
    }

    CELER_ENSURE(world_);
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
} // namespace celeritas
