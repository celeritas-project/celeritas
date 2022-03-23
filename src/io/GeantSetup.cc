//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeantSetup.cc
//---------------------------------------------------------------------------//
#include "GeantSetup.hh"

#include "detail/GeantVersion.hh"
#if CELERITAS_G4_V10
#    include <G4RunManager.hh>
#else
#    include <G4RunManagerFactory.hh>
#endif

#include <FTFP_BERT.hh>
#include <G4GenericPhysicsList.hh>
#include <G4UImanager.hh>
#include <G4VModularPhysicsList.hh>

#include "base/ScopedTimeAndRedirect.hh"

#include "detail/ActionInitialization.hh"
#include "detail/DetectorConstruction.hh"
#include "detail/GeantExceptionHandler.hh"
#include "detail/GeantLoggerAdapter.hh"
#include "detail/GeantPhysicsList.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from a GDML file and physics options.
 */
GeantSetup::GeantSetup(const std::string& gdml_filename, PhysicsList physics)
{
    CELER_LOG(status) << "Initializing Geant4";

    {
        // Run manager writes output that cannot be redirected...
        ScopedTimeAndRedirect scoped_time("G4RunManager");
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

    //// Initialize the geometry ////

    auto detector
        = std::make_unique<detail::DetectorConstruction>(gdml_filename);
    // Get world_volume for store_geometry() before releasing detector ptr
    world_ = detector->get_world_volume();
    run_manager_->SetUserInitialization(detector.release());

    //// Load the physics list ////

    std::unique_ptr<G4VUserPhysicsList> physics_list;

    switch (physics)
    {
        case PhysicsList::em_basic:
            physics_list = std::make_unique<detail::GeantPhysicsList>();
            break;
        case PhysicsList::em_standard: {
            auto physics_constructor
                = std::make_unique<std::vector<G4String>>();
            physics_constructor->push_back("G4EmStandardPhysics");
            physics_list = std::make_unique<G4GenericPhysicsList>(
                physics_constructor.release());
            break;
        }
        case PhysicsList::ftfp_bert:
            // Full Physics
            physics_list = std::make_unique<FTFP_BERT>();
            break;
    }
    run_manager_->SetUserInitialization(physics_list.release());

    // Minimal run to generate the physics tables
    auto action_initialization
        = std::make_unique<detail::ActionInitialization>();
    run_manager_->SetUserInitialization(action_initialization.release());
    G4UImanager::GetUIpointer()->ApplyCommand("/run/initialize");
    run_manager_->BeamOn(1);

    CELER_ENSURE(*this);
    CELER_ENSURE(world_);
}

//---------------------------------------------------------------------------//
/*!
 * Delete a geant4 run manager.
 */
void GeantSetup::RMDeleter::operator()(G4RunManager* rm) const
{
    delete rm;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
