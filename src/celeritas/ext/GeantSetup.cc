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

#include "detail/GeantVersion.hh"
#if CELERITAS_G4_V10
#    include <G4RunManager.hh>
#else
#    include <G4RunManagerFactory.hh>
#endif

#include "corecel/io/ScopedTimeAndRedirect.hh"

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
/*!
 * Create a particle gun and generate one primary for a minimal simulation run.
 */
class PrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction
{
  public:
    PrimaryGeneratorAction();

    //! Generate a priary at the beginning of each event
    void GeneratePrimaries(G4Event* event) override
    {
        CELER_EXPECT(particle_gun_);
        particle_gun_->GeneratePrimaryVertex(event);
    }

  private:
    std::unique_ptr<G4ParticleGun> particle_gun_;
};

//---------------------------------------------------------------------------//
/*!
 * Construct a particle gun for the minimal simulation run.
 */
PrimaryGeneratorAction::PrimaryGeneratorAction() : particle_gun_(nullptr)
{
    // Select particle type
    G4ParticleDefinition* particle;
    particle = G4ParticleTable::GetParticleTable()->FindParticle("e-");
    CELER_ASSERT(particle);

    // Create and set up particle gun
    const int number_of_particles = 1;
    particle_gun_ = std::make_unique<G4ParticleGun>(number_of_particles);
    particle_gun_->SetParticleDefinition(particle);
    particle_gun_->SetParticleMomentumDirection(G4ThreeVector(0., 0., 1.));
    particle_gun_->SetParticleEnergy(10 * GeV);
    particle_gun_->SetParticlePosition(G4ThreeVector(0, 0, 0));
}

//---------------------------------------------------------------------------//
/*!
 * Initialize Geant4.
 */
class ActionInitialization : public G4VUserActionInitialization
{
  public:
    void Build() const override
    {
        auto action = std::make_unique<PrimaryGeneratorAction>();
        this->SetUserAction(action.release());
    }
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
        run_manager_ = std::make_unique<G4RunManager>();
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
        auto init = std::make_unique<ActionInitialization>();
        run_manager_->SetUserInitialization(init.release());
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
