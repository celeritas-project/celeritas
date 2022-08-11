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

#include <FTFP_BERT.hh>
#include <G4EmParameters.hh>
#include <G4GenericPhysicsList.hh>
#include <G4ParticleTable.hh>
#include <G4VModularPhysicsList.hh>
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
 * Set up global geant4 physics.
 */
void load_physics(const GeantSetupOptions& options, G4RunManager* run_manager)
{
    std::unique_ptr<G4VUserPhysicsList> physics_list;

    using PL = GeantSetupPhysicsList;
    switch (options.physics)
    {
        case PL::none:
            // Do not load any physics (possibly for geometry-only testing or
            // visualization)
            return;
        case PL::em_basic:
            physics_list = std::make_unique<detail::GeantPhysicsList>();
            break;
        case PL::em_standard: {
            auto physics_constructor
                = std::make_unique<std::vector<G4String>>();
            physics_constructor->push_back("G4EmStandardPhysics");
            physics_list = std::make_unique<G4GenericPhysicsList>(
                physics_constructor.release());
            break;
        }
        case PL::ftfp_bert:
            // Full Physics
            physics_list = std::make_unique<FTFP_BERT>();
            break;
        default:
            CELER_VALIDATE(false, << "invalid physics list");
    }

    {
        // Set EM options
        auto& em_parameters = *G4EmParameters::Instance();
        CELER_VALIDATE(options.em_bins_per_decade > 0,
                       << "number of EM bins per decade="
                       << options.em_bins_per_decade << " (must be positive)");
        em_parameters.SetNumberOfBinsPerDecade(options.em_bins_per_decade);
    }

    CELER_ASSERT(physics_list);
    run_manager->SetUserInitialization(physics_list.release());
}

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
    load_physics(options, run_manager_.get());

    // Generate physics tables
    if (options.physics != GeantSetupPhysicsList::none)
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
