//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/TrackingManagerIntegration.test.cc
//---------------------------------------------------------------------------//
#include "accel/TrackingManagerIntegration.hh"

#include <memory>
#include <G4EmStandardPhysics.hh>
#include <G4RunManager.hh>
#include <G4VUserActionInitialization.hh>

#include "IntegrationTestBase.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// Helper class for setting up Celeritas
class ActionInitialization final : public G4VUserActionInitialization
{
  public:
    void BuildForMaster() const final
    {
        simple_offload.BuildForMaster(&setup_options, &shared_params);

        CELER_LOG_LOCAL(status) << "Constructing user actions";

        this->SetUserAction(new RunAction{});
    }
    void Build() const final
    {
        simple_offload.Build(
            &setup_options, &shared_params, &local_transporter);

        CELER_LOG_LOCAL(status) << "Constructing user actions";

        this->SetUserAction(new PrimaryGeneratorAction{});
        this->SetUserAction(new RunAction{});
        this->SetUserAction(new EventAction{});
    }
};

//---------------------------------------------------------------------------//

class TrackingManagerIntegrationTest : public IntegrationTestBase
{
    using Base = IntegrationTestBase;

  protected:
    static void SetUpTestCase();
};

void TrackingManagerIntegrationTest::SetUpTestCase()
{
    G4RunManager& run_manager = Base::run_manager();

    // Set up geometry
    run_manager.SetUserInitialization(
        Base::make_detector_construction().release());

    // Set up physics
    auto physics_list
        = std::make_unique<G4EmStandardPhysics>(/* verbosity = */ 0);
    run_manager.SetUserInitialization(physics_list.release());

    auto action_init = std::make_unique<ActionInitialization>();
    run_manager->SetUserInitialization(action_init.release());
}

//---------------------------------------------------------------------------//

TEST_F(TrackingManagerIntegrationTest, initialize)
{
    Base::run_manager().Initialize();
}

TEST_F(TrackingManagerIntegrationTest, run)
{
    Base::run_manager().BeamOn(2);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
