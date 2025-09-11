//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/TrackingManagerIntegration.test.cc
//---------------------------------------------------------------------------//
#include "accel/TrackingManagerIntegration.hh"

#include <atomic>
#include <functional>
#include <G4RunManager.hh>
#include <G4Threading.hh>
#include <G4UImanager.hh>
#include <G4VModularPhysicsList.hh>

#include "corecel/io/Logger.hh"
#include "geocel/GeantUtils.hh"
#include "celeritas/global/CoreState.hh"
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
namespace
{
//! Query thread-local data to determine whether the thread is running
bool is_running_events()
{
    return !G4Threading::IsMasterThread()
           || !G4Threading::IsMultithreadedApplication();
}

}  // namespace

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
        if (check_during_run_)
        {
            check_during_run_();
        }
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

    std::function<void()> check_during_run_;
};

//---------------------------------------------------------------------------//
class LarSphere : public LarSphereIntegrationMixin, public TMITestBase
{
};

/*!
 * Check that multiple sequential runs complete successfully.
 */
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

/*!
 * Check that UI commands are correctly propagated to the Celeritas runtime.
 */
TEST_F(LarSphere, ui)
{
    auto& rm = this->run_manager();
    auto& tmi = TMI::Instance();

    EXPECT_EQ(tmi.GetMode(), OffloadMode::uninitialized);
    tmi.SetOptions(this->make_setup_options());
    EXPECT_NE(tmi.GetMode(), OffloadMode::uninitialized);

    std::atomic<int> check_count{0};

    auto& ui = *G4UImanager::GetUIpointer();
    if (SharedParams::GetMode() != OffloadMode::disabled)
    {
        ui.ApplyCommand("/celer/maxNumTracks 128");
        ui.ApplyCommand("/celer/maxInitializers 10000");

        check_during_run_ = [&check_count, &tmi] {
            EXPECT_NE(OffloadMode::uninitialized, tmi.GetMode());

            if (tmi.GetMode() == OffloadMode::enabled && is_running_events())
            {
                CELER_LOG_LOCAL(debug) << "Checking number of tracks";
                ++check_count;

                auto const& state = tmi.GetState();
                EXPECT_EQ(state.size(), 128);
            }
        };
    }
    else
    {
        check_during_run_ = [&check_count] {
            if (is_running_events())
            {
                ++check_count;
            }
        };
    }

    ui.ApplyCommand("/run/initialize");
    ui.ApplyCommand("/run/beamOn 2");

    EXPECT_EQ(get_geant_num_threads(rm), check_count.load());
}

//---------------------------------------------------------------------------//
class TestEm3 : public TestEm3IntegrationMixin, public TMITestBase
{
};

/*!
 * Check that TestEm3 runs.
 */
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
