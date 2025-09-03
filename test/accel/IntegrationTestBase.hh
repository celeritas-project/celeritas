//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/IntegrationTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string_view>

#include "celeritas/ext/GeantPhysicsOptions.hh"
#include "celeritas/inp/Events.hh"

#include "Test.hh"

class G4RunManager;
class G4Run;
class G4UserSteppingAction;
class G4UserTrackingAction;
class G4Event;
class G4VModularPhysicsList;
class G4VSensitiveDetector;

namespace celeritas
{
struct SetupOptions;
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Help set up Geant4 for integration testing.
 *
 * Calling setup_run_manager will:
 * - Create G4RunManager using the factory
 * - Create and set the detector construction, calling \c gdml_basename on the
 *   main thread to load the detector
 * - Create and set the physics list by calling \c make_physics_list
 * - Create and set the internal action initialization
 *
 * The detector construction will:
 * - Load the GDML file on the main thread
 * - Call \c make_sens_det on each worker thread
 *
 * The action initialization will create several classes which can dispatch
 * back to the test harness:
 * - Create the primary generator which uses \c make_primary_input
 * - Create the event and run actions, which call virtual functions in the test
 *   harness
 * - Optionally create and attach tracking/stepping managers
 *
 * The run manager will be deleted when the GoogleTest harness is torn down.
 */
class IntegrationTestBase : public ::celeritas::test::Test
{
  public:
    //!@{
    //! \name Type aliases
    using PrimaryInput = celeritas::inp::CorePrimaryGenerator;
    using PhysicsInput = celeritas::GeantPhysicsOptions;
    using UPPhysicsList = std::unique_ptr<G4VModularPhysicsList>;
    using UPTrackAction = std::unique_ptr<G4UserTrackingAction>;
    using UPStepAction = std::unique_ptr<G4UserSteppingAction>;
    using UPSensDet = std::unique_ptr<G4VSensitiveDetector>;
    //!@}

  public:
    // Default destructor to enable base class deletion and anchor vtable
    virtual ~IntegrationTestBase();

    // Lazily create and/or access the run manager
    G4RunManager& run_manager();

    //! Set the GDML filename (in test/geocel/data without ".gdml")
    virtual std::string_view gdml_basename() const = 0;

    //! Create options for the primary generator
    virtual PrimaryInput make_primary_input() const = 0;

    // Create options for EM physics setup
    virtual PhysicsInput make_physics_input() const;

    // Create physics list: default is EM only using make_physics_input
    virtual UPPhysicsList make_physics_list() const;

    // Create optional tracking action (local, default null)
    virtual UPTrackAction make_tracking_action();

    // Create optional stepping action (local, default null)
    virtual UPStepAction make_stepping_action();

    // Create Celeritas setup options
    virtual SetupOptions make_setup_options();

    // Create THREAD-LOCAL sensitive detectors for an SD name in the GDML file
    virtual UPSensDet make_sens_det(std::string const& sd_name);

    //!@{
    //! \name Dispatch from user run/event actions
    virtual void BeginOfRunAction(G4Run const* run) = 0;
    virtual void EndOfRunAction(G4Run const* run) = 0;
    virtual void BeginOfEventAction(G4Event const* event) = 0;
    virtual void EndOfEventAction(G4Event const* event) = 0;
    //!@}
};

//---------------------------------------------------------------------------//
//! Generate LAr sphere geometry with 10 MeV electrons
class LarSphereIntegrationMixin : virtual public IntegrationTestBase
{
    using Base = IntegrationTestBase;

  public:
    std::string_view gdml_basename() const final { return "lar-sphere"; }
    PrimaryInput make_primary_input() const override;
    PhysicsInput make_physics_input() const override;
    UPSensDet make_sens_det(std::string const&) override;
};

//---------------------------------------------------------------------------//
//! Generate TestEM3 geometry with 100 MeV electrons
class TestEm3IntegrationMixin : virtual public IntegrationTestBase
{
    using Base = IntegrationTestBase;

  public:
    std::string_view gdml_basename() const final { return "testem3"; }
    PrimaryInput make_primary_input() const override;
    PhysicsInput make_physics_input() const override;
    UPSensDet make_sens_det(std::string const&) override;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
