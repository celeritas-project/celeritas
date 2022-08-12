//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantSetup.cc
//---------------------------------------------------------------------------//
#include "GeantSetup.hh"

#include "detail/GeantVersion.hh"
#if CELERITAS_G4_V10
#    include <G4RunManager.hh>
#else
#    include <G4RunManagerFactory.hh>
#endif

#include <G4ParticleTable.hh>
#include <G4VUserDetectorConstruction.hh>

#include "corecel/io/ScopedTimeAndRedirect.hh"

#include "LoadGdml.hh"
#include "detail/ActionInitialization.hh"
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
    // Construct from a GDML filename
    explicit DetectorConstruction(const std::string& filename)
    {
        phys_vol_world_ = load_gdml(filename);
        CELER_ENSURE(phys_vol_world_);
    }

    G4VPhysicalVolume* Construct() override
    {
        CELER_EXPECT(phys_vol_world_);
        return phys_vol_world_.release();
    }

    const G4VPhysicalVolume* world_volume() const
    {
        CELER_EXPECT(phys_vol_world_);
        return phys_vol_world_.get();
    }

  private:
    UPG4PhysicalVolume phys_vol_world_;
};

//---------------------------------------------------------------------------//
} // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct from a GDML file and physics options.
 */
GeantSetup::GeantSetup(const std::string& gdml_filename, Options options)
{
    CELER_LOG(status) << "Initializing Geant4";

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
        run_manager_.reset(new G4RunManager);
#else
        run_manager_.reset(
            G4RunManagerFactory::CreateRunManager(G4RunManagerType::Serial));
#endif
        CELER_ASSERT(run_manager_);
    }

    detail::GeantLoggerAdapter    scoped_logger;
    detail::GeantExceptionHandler scoped_exception_handler;

    // Initialize geometry
    {
        auto detector = std::make_unique<DetectorConstruction>(gdml_filename);

        // Get world_volume for store_geometry() before releasing detector ptr
        world_ = detector->world_volume();

        run_manager_->SetUserInitialization(detector.release());
    }

    // Construct the physics
    {
        auto physics_list = std::make_unique<detail::GeantPhysicsList>(options);
        run_manager_->SetUserInitialization(physics_list.release());
    }

    // Generate physics tables
    {
        auto action_initialization
            = std::make_unique<detail::ActionInitialization>();
        run_manager_->SetUserInitialization(action_initialization.release());
        run_manager_->Initialize();
        run_manager_->BeamOn(1);
    }

    CELER_ENSURE(world_);
    CELER_ENSURE(*this);
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
