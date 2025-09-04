//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/TrackingManagerIntegration.test.cc
//---------------------------------------------------------------------------//
#include "accel/TrackingManagerIntegration.hh"

#include <G4RunManager.hh>
#include <G4VModularPhysicsList.hh>

#include "corecel/io/Logger.hh"
#include "accel/SetupOptions.hh"
#include "accel/TrackingManagerConstructor.hh"
#include "accel/detail/IntegrationSingleton.hh"

#include "IntegrationTestBase.hh"
#include "celeritas_test.hh"

using TMI = celeritas::TrackingManagerIntegration;

namespace celeritas
{
namespace test
{

//---------------------------------------------------------------------------//
/*!
 * Test the TrackingManagerIntegration (TMI).
 *
 * The tracking manager will:
 * - Add a physics constructor that sets up tracking managers for the supported
 *   particles
 * - Set up Celeritas shared data at BeginOfRunAction on the main thread
 * - Set up Celeritas local data at BeginOfRunAction on the worker thread
 * - Clean up on EndOfRunAction
 */
class TMITestBase : virtual public IntegrationTestBase
{
    using Base = IntegrationTestBase;

  protected:
    UPPhysicsList make_physics_list() const override
    {
        auto physics = Base::make_physics_list();
        CELER_ASSERT(physics);
        physics->RegisterPhysics(
            new TrackingManagerConstructor(&TMI::Instance()));
        return physics;
    }
    void BeginOfRunAction(G4Run const* run) override
    {
        TMI::Instance().BeginOfRunAction(run);
    }
    void EndOfRunAction(G4Run const* run) override
    {
        TMI::Instance().EndOfRunAction(run);
    }
    void BeginOfEventAction(G4Event const*) override {}
    void EndOfEventAction(G4Event const*) override
    {
        auto const& local_transport
            = detail::IntegrationSingleton::local_transporter();
        EXPECT_EQ(0, local_transport.GetBufferSize());
    }
};

//---------------------------------------------------------------------------//
class LarSphere : public LarSphereIntegrationMixin, public TMITestBase
{
};

TEST_F(LarSphere, run)
{
    auto& rm = this->run_manager();
    TMI::Instance().SetOptions(this->make_setup_options());

    CELER_LOG(status) << "Run initialization";
    rm.Initialize();

    CELER_LOG(status) << "Beam on (first run)";
    rm.BeamOn(3);

    CELER_LOG(status) << "Beam on (second run)";
    rm.BeamOn(1);
}

//---------------------------------------------------------------------------//
class TestEm3 : public TestEm3IntegrationMixin, public TMITestBase
{
};

TEST_F(TestEm3, run)
{
    auto& rm = this->run_manager();
    TMI::Instance().SetOptions(this->make_setup_options());

    CELER_LOG(status) << "Run initialization";
    rm.Initialize();

    CELER_LOG(status) << "Beam on (first run)";
    rm.BeamOn(2);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
