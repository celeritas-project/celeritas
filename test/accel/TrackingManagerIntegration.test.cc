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
#include <G4UserTrackingAction.hh>
#include <G4VModularPhysicsList.hh>

#include "corecel/io/Logger.hh"
#include "geocel/GeantUtils.hh"
#include "geocel/UnitUtils.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/optical/OpticalCollector.hh"
#include "accel/LocalTransporter.hh"
#include "accel/SetupOptions.hh"
#include "accel/SharedParams.hh"
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
// TEST BASE
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
    void BeginOfEventAction(G4Event const* event) override
    {
        if (event->GetEventID() == 1)
        {
            for (auto i : range(event->GetNumberOfPrimaryVertex()))
            {
                G4PrimaryVertex* vtx = event->GetPrimaryVertex(i);
                for (auto j : range(vtx->GetNumberOfParticle()))
                {
                    G4PrimaryParticle* p = vtx->GetPrimary(j);
                    p->SetWeight(10.0);
                }
            }
        }
    }

    virtual void process_hit(G4Step const* step) override
    {
        LarSphereIntegrationMixin::process_hit(step);
        ASSERT_TRUE(step);

        // Check the weight is consistent with our modification at
        // begin-of-event
        auto event_id = G4EventManager::GetEventManager()
                            ->GetConstCurrentEvent()
                            ->GetEventID();
        EXPECT_DOUBLE_EQ((event_id == 1 ? 10.0 : 1.0),
                         step->GetTrack()->GetWeight());
    }
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

    if (this->HasFailure())
    {
        GTEST_SKIP() << "Skipping remaining tests since we've already failed";
    }

    CELER_LOG(status) << "Beam on (second run)";
    rm.BeamOn(1);
}

/*!
 * Check that UI commands are correctly propagated to the Celeritas runtime.
 */
TEST_F(LarSphere, run_ui)
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
// LAR SPHERE WITH OPTICAL
//---------------------------------------------------------------------------//
class TrackingAction : public G4UserTrackingAction
{
  public:
    void PreUserTrackingAction(G4Track const* t)
    {
        if (t->GetParticleDefinition()->GetPDGEncoding() == -22)
        {
            ++counter_;
        }
    }
    std::size_t num_photons() const { return counter_; }

  private:
    std::size_t counter_{};
};

/*!
 * Test the LarSphere, offloading both EM tracks *and* optical photons.
 */
class LarSphereOptical : public LarSphere
{
  public:
    PhysicsInput make_physics_input() const override;
    PrimaryInput make_primary_input() const override;
    SetupOptions make_setup_options() override;
    void EndOfRunAction(G4Run const* run) override;
    UPTrackAction make_tracking_action() override
    {
        auto result = std::make_unique<TrackingAction>();
        {
            // Store the raw pointer in the tracking_ vector using a static
            // mutex
            static std::mutex mutex;
            std::lock_guard<std::mutex> lock(mutex);
            tracking_.push_back(result.get());
        }
        return result;
    }

  private:
    std::vector<TrackingAction*> tracking_;
};

//---------------------------------------------------------------------------//
/*!
 * Enable optical physics.
 */
auto LarSphereOptical::make_physics_input() const -> PhysicsInput
{
    auto result = LarSphereIntegrationMixin::make_physics_input();

    // Set default optical physics
    auto& optical = result.optical;
    optical = {};
    EXPECT_TRUE(optical);

    // Disable WLS which isn't yet working (reemission) in Celeritas
    using WLSO = WavelengthShiftingOptions;
    optical.wavelength_shifting = WLSO::deactivated();
    optical.wavelength_shifting2 = WLSO::deactivated();

    return result;
}

auto LarSphereOptical::make_primary_input() const -> PrimaryInput
{
    auto result = LarSphereIntegrationMixin::make_primary_input();

    result.shape = inp::PointDistribution{from_cm({0.1, 0.1, 0})};
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Enable optical tracking.
 */
auto LarSphereOptical::make_setup_options() -> SetupOptions
{
    auto result = LarSphereIntegrationMixin::make_setup_options();

    result.optical_capacity = [] {
        inp::OpticalStateCapacity cap;
        cap.primaries = 32768;
        cap.tracks = 4096;
        cap.generators = 2048;
        return cap;
    }();

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Test that the optical tracking loop completed correctly.
 *
 * - Generator counters show whether any photons are queued but not run
 * - Accumulated stats show whether the state has run some photons
 */
void LarSphereOptical::EndOfRunAction(G4Run const* run)
{
    auto& integration = detail::IntegrationSingleton::instance();
    if (integration.mode() == OffloadMode::enabled)
    {
        auto& local_transporter = integration.local_transporter();
        auto const& shared_params = integration.shared_params();

        // Check that local/shared data is available before end of run
        EXPECT_EQ(is_running_events(), static_cast<bool>(local_transporter));
        EXPECT_TRUE(shared_params) << "Celeritas was not enabled";

        auto const& optical_collector = shared_params.optical();
        EXPECT_TRUE(optical_collector) << "optical offloading was not enabled";
        if (local_transporter && optical_collector)
        {
            // Use diagnostic methods to check counters
            auto& aux_state = local_transporter.GetState().aux();
            auto accum_stats = optical_collector->exchange_counters(aux_state);
            CELER_LOG_LOCAL(info)
                << "Ran " << accum_stats.steps << " over "
                << accum_stats.step_iters << " step iterations from "
                << accum_stats.flushes << " flushes";
            EXPECT_GT(accum_stats.steps, 0);
            EXPECT_GT(accum_stats.step_iters, 0);
            EXPECT_GT(accum_stats.flushes, 0);

            auto counts = optical_collector->buffer_counts(aux_state);
            EXPECT_EQ(0, counts.buffer_size);  //!< Pending generators
            EXPECT_EQ(0, counts.num_pending);  //!< Photons pending generation
            EXPECT_EQ(0, counts.num_generated);  //!< Photons generated
        }
    }
    if (G4Threading::IsMasterThread())
    {
        std::size_t num_photons_generated{0};
        for (auto* tracking_action : tracking_)
        {
            num_photons_generated += tracking_action->num_photons();
        }
        CELER_LOG(info) << "Processed a total of " << num_photons_generated
                        << " photons";

        if (integration.mode() == OffloadMode::enabled)
        {
            EXPECT_EQ(0, num_photons_generated);
        }
        else
        {
            EXPECT_GT(num_photons_generated, 0);
        }
    }

    // Continue cleanup and other checks at end of run
    LarSphere::EndOfRunAction(run);
}

//---------------------------------------------------------------------------//
/*!
 * Check that the test runs.
 */
TEST_F(LarSphereOptical, run)
{
    auto& rm = this->run_manager();
    TMI::Instance().SetOptions(this->make_setup_options());

    CELER_LOG(status) << "Run initialization";
    rm.Initialize();
    CELER_LOG(status) << "Run two events";
    rm.BeamOn(2);

    if (this->HasFailure())
    {
        GTEST_SKIP() << "Skipping remaining tests since we've already failed";
    }
    CELER_LOG(status) << "Run one more event";
    rm.BeamOn(2);
}

//---------------------------------------------------------------------------//
// TESTEM3
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

    if (this->HasFailure())
    {
        GTEST_SKIP() << "Skipping remaining tests since we've already failed";
    }

    CELER_LOG(status) << "Beam on (first run)";
    rm.BeamOn(2);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
