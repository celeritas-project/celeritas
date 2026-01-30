//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/TrackingManagerIntegration.test.cc
//---------------------------------------------------------------------------//
#include "accel/TrackingManagerIntegration.hh"

#include <atomic>
#include <functional>
#include <mutex>
#include <regex>
#include <string_view>
#include <G4RunManager.hh>
#include <G4Threading.hh>
#include <G4UImanager.hh>
#include <G4UserTrackingAction.hh>
#include <G4VModularPhysicsList.hh>

#include "corecel/StringSimplifier.hh"
#include "corecel/cont/Array.hh"
#include "corecel/io/Logger.hh"
#include "geocel/GeantUtils.hh"
#include "geocel/UnitUtils.hh"
#include "celeritas/ext/GeantParticleView.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/inp/Events.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/OpticalCollector.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "accel/LocalTransporter.hh"
#include "accel/SetupOptions.hh"
#include "accel/SharedParams.hh"
#include "accel/TrackingManagerConstructor.hh"
#include "accel/detail/IntegrationSingleton.hh"

#include "IntegrationTestBase.hh"
#include "TestMacros.hh"
#include "celeritas_test.hh"

using TMI = celeritas::TrackingManagerIntegration;
using namespace std::string_view_literals;

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

constexpr bool using_surface_vg = CELERITAS_VECGEOM_SURFACE
                                  && CELERITAS_CORE_GEO
                                         == CELERITAS_CORE_GEO_VECGEOM;

/*!
 * Count particle types.
 */
class CounterTrackingAction final : public G4UserTrackingAction
{
  public:
    void PreUserTrackingAction(G4Track const* t) final
    {
        GeantParticleView particle{*t->GetParticleDefinition()};

        if (particle.pdg() == pdg::electron())
        {
            ++num_electrons_;
        }
        if (particle.pdg() == pdg::positron())
        {
            ++num_positrons_;
        }
        else if (particle.is_optical_photon())
        {
            ++num_photons_;
        }
    }
    std::size_t num_photons() const { return num_photons_; }
    std::size_t num_electrons() const { return num_electrons_; }
    std::size_t num_positrons() const { return num_positrons_; }

  private:
    std::size_t num_photons_{};
    std::size_t num_electrons_{};
    std::size_t num_positrons_{};
};

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
// LAR SPHERE (EM only)
//---------------------------------------------------------------------------//
class LarSphere : public LarSphereIntegrationMixin, public TMITestBase
{
  public:
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

    //! Check wrapped RuntimeError caught by GeantExceptionHandler
    void caught_g4_runtime_error(RuntimeError const& e) override
    {
        if (!check_runtime_errors_)
        {
            // Let the base class manage and fail on the caught error
            return TMITestBase::caught_g4_runtime_error(e);
        }
        CELER_EXPECT(std::string_view(e.details().which) == "Geant4"sv);

        static std::recursive_mutex exc_mutex;
        std::lock_guard scoped_lock{exc_mutex};

        static std::regex extract_error{R"(runtime error:\s*(.+?)(?:\n|$))"};
        std::smatch match;
        std::string what = e.what();
        if (std::regex_search(what, match, extract_error))
        {
            CELER_ASSERT(match.size() > 1);
            exceptions_.push_back(match[1].str());
        }
        else
        {
            exceptions_.push_back(std::move(what));
        }
    }

    void TearDown() override
    {
        if (!exceptions_.empty())
        {
            FAIL() << exceptions_.size()
                   << " runtime errors were caught but not checked";
        }
    }

    //! Append caught exceptions in this local test rather than failing
    bool check_runtime_errors_{false};
    //! Exceptions that were caught by this test suite's error handler
    std::vector<std::string> exceptions_;
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
    if (using_surface_vg)
    {
        GTEST_SKIP() << "VecGeom surface model does not support multiple runs";
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

/*!
 * Check that omitting the SetOptions call causes the expected errors.
 */
TEST_F(LarSphere, no_set_options)
{
    auto& rm = this->run_manager();
    TMI::Instance();
    check_runtime_errors_ = true;

    CELER_LOG(status) << "Run initialization";
    rm.Initialize();
    EXPECT_EQ(0, exceptions_.size());
    CELER_LOG(status) << "Run two events";
    rm.BeamOn(2);

    std::vector<std::string> expected_exceptions = {
        "SetOptions or UI entries were not completely set before BeginRun",
    };
    if (!G4Threading::IsMultithreadedApplication())
    {
        // Geant4 still starts the first local event if an error happens during
        // BeginOfRun
        expected_exceptions.push_back(
            R"(Celeritas was not initialized properly (maybe BeginOfRunAction was not called?))");
    }
    EXPECT_VEC_EQ(expected_exceptions, exceptions_);
    exceptions_.clear();
}

//---------------------------------------------------------------------------//
// LAR SPHERE WITH OPTICAL
//---------------------------------------------------------------------------//
/*!
 * Test the LarSphere, offloading both EM tracks *and* optical photons.
 */
class LarSphereOptical : public LarSphere
{
  public:
    PhysicsInput make_physics_input() const override
    {
        auto result = LarSphere::make_physics_input();
        enable_optical_physics(result);
        return result;
    }

    PrimaryInput make_primary_input() const override
    {
        auto result = LarSphereIntegrationMixin::make_primary_input();

        result.shape = inp::PointDistribution{
            array_cast<double>(from_cm({0.1, 0.1, 0}))};
        result.primaries_per_event = 1;
        result.energy = inp::MonoenergeticDistribution{2};  // [MeV]
        return result;
    }

    SetupOptions make_setup_options() override;

    void EndOfRunAction(G4Run const* run) override;

    UPTrackAction make_tracking_action() override
    {
        auto result = std::make_unique<CounterTrackingAction>();
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
    std::vector<CounterTrackingAction*> tracking_;
};

//---------------------------------------------------------------------------//
/*!
 * Enable optical tracking.
 */
auto LarSphereOptical::make_setup_options() -> SetupOptions
{
    auto result = LarSphereIntegrationMixin::make_setup_options();

    result.optical = [] {
        OpticalSetupOptions opt;
        opt.capacity.tracks = 32768;
        opt.capacity.generators = 32768 * 8;
        opt.capacity.primaries = opt.capacity.generators;
        return opt;
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

        auto const& optical_collector
            = shared_params.problem_loaded().optical_collector;
        EXPECT_TRUE(optical_collector) << "optical offloading was not enabled";
        if (local_transporter && optical_collector)
        {
            // Use diagnostic methods to check counters
            auto const& accum_stats
                = optical_collector->optical_state(local_transporter.GetState())
                      .accum();
            CELER_LOG_LOCAL(info)
                << "Ran " << accum_stats.steps << " over "
                << accum_stats.step_iters << " step iterations from "
                << accum_stats.flushes << " flushes";
            EXPECT_GT(accum_stats.steps, 0);
            EXPECT_GT(accum_stats.step_iters, 0);
            EXPECT_GT(accum_stats.flushes, 0);

            auto& aux_state = local_transporter.GetState().aux();
            auto counts = optical_collector->buffer_counts(aux_state);
            EXPECT_EQ(0, counts.buffer_size);  //!< Pending generators
            EXPECT_EQ(0, counts.num_pending);  //!< Photons pending generation
            EXPECT_EQ(0, counts.num_generated);  //!< Photons generated
        }
    }
    if (G4Threading::IsMasterThread())
    {
        std::size_t photons{0};
        std::size_t electrons{0};
        for (auto* tracking_action : tracking_)
        {
            photons += tracking_action->num_photons();
            electrons += tracking_action->num_electrons();
        }
        CELER_LOG(info) << "Geant4 tracked a total of " << photons
                        << " optical photons"
                        << " and " << electrons << " electrons";

        if (integration.mode() == OffloadMode::enabled)
        {
            EXPECT_EQ(0, photons);
            EXPECT_EQ(0, electrons);
        }
        else
        {
            EXPECT_GT(photons, 0);
            EXPECT_GT(electrons, 0);
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
}

//---------------------------------------------------------------------------//
// OPNOVICE
//---------------------------------------------------------------------------//
/*!
 * Test the Op-Novice example, offloading optical photons.
 */
class OpNoviceOptical : public OpNoviceIntegrationMixin, public TMITestBase
{
  public:
    void EndOfRunAction(G4Run const* run) override;
    UPTrackAction make_tracking_action() override
    {
        auto result = std::make_unique<CounterTrackingAction>();
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
    std::vector<CounterTrackingAction*> tracking_;
};

//---------------------------------------------------------------------------//
/*!
 * Test that the optical tracking loop completed correctly.
 *
 * - Generator counters show whether any photons are queued but not run
 * - Accumulated stats show whether the state has run some photons
 */
void OpNoviceOptical::EndOfRunAction(G4Run const* run)
{
    auto& integration = detail::IntegrationSingleton::instance();
    if (integration.mode() == OffloadMode::enabled)
    {
        auto& local_transporter = integration.local_transporter();
        auto const& shared_params = integration.shared_params();

        // Check that local/shared data is available before end of run
        EXPECT_EQ(is_running_events(), static_cast<bool>(local_transporter));
        EXPECT_TRUE(shared_params) << "Celeritas was not enabled";

        auto const& optical_collector
            = shared_params.problem_loaded().optical_collector;
        EXPECT_TRUE(optical_collector) << "optical offloading was not enabled";
        if (local_transporter && optical_collector)
        {
            // Use diagnostic methods to check counters
            auto const& accum_stats
                = optical_collector->optical_state(local_transporter.GetState())
                      .accum();
            CELER_LOG_LOCAL(info)
                << "Ran " << accum_stats.steps << " over "
                << accum_stats.step_iters << " step iterations from "
                << accum_stats.flushes << " flushes";
            EXPECT_GT(accum_stats.steps, 0);
            EXPECT_GT(accum_stats.step_iters, 0);
            EXPECT_GT(accum_stats.flushes, 0);

            auto& aux_state = local_transporter.GetState().aux();
            auto counts = optical_collector->buffer_counts(aux_state);
            EXPECT_EQ(0, counts.buffer_size);  //!< Pending generators
            EXPECT_EQ(0, counts.num_pending);  //!< Photons pending generation
            EXPECT_EQ(0, counts.num_generated);  //!< Photons generated
        }
    }
    if (G4Threading::IsMasterThread())
    {
        std::size_t photons{0};
        std::size_t positrons{0};
        for (auto* tracking_action : tracking_)
        {
            photons += tracking_action->num_photons();
            positrons += tracking_action->num_positrons();
        }
        CELER_LOG(info) << "Geant4 tracked a total of " << photons
                        << " optical photons"
                        << " and " << positrons << " positrons";

        if (integration.mode() == OffloadMode::enabled)
        {
            EXPECT_EQ(0, photons);
            EXPECT_EQ(0, positrons);
        }
        else
        {
            EXPECT_GT(photons, 0);
            EXPECT_GT(positrons, 0);
        }
    }

    // Continue cleanup and other checks at end of run
    TMITestBase::EndOfRunAction(run);
}

//---------------------------------------------------------------------------//
/*!
 * Check that the OpNovice test run.
 */
TEST_F(OpNoviceOptical, run)
{
    auto& rm = this->run_manager();
    TMI::Instance().SetOptions(this->make_setup_options());

    CELER_LOG(status) << "Run initialization";
    rm.Initialize();
    CELER_LOG(status) << "Run two events";
    rm.BeamOn(10);
}

//---------------------------------------------------------------------------//
// OPTICAL SURFACES
//---------------------------------------------------------------------------//
/*!
 * Test the LarSphere, offloading both EM tracks *and* optical photons.
 */
class OpticalSurfaces : public TMITestBase
{
  public:
    std::string_view gdml_basename() const final { return "optical-surfaces"; }
    PhysicsInput make_physics_input() const override
    {
        auto result = TMITestBase::make_physics_input();
        enable_optical_physics(result);
        return result;
    }

    PrimaryInput make_primary_input() const override;
    SetupOptions make_setup_options() override;
    void EndOfRunAction(G4Run const* run) override;
};

//---------------------------------------------------------------------------//
/*!
 * Fire positrons through the liquid argon toward the detectors.
 */
auto OpticalSurfaces::make_primary_input() const -> PrimaryInput
{
    PrimaryInput result;
    result.pdg = {pdg::positron()};
    result.shape
        = inp::PointDistribution{array_cast<double>(from_cm({30, 0, 0}))};
    result.angle = inp::MonodirectionalDistribution{{-1, 0, 0}};
    result.energy = inp::MonoenergeticDistribution{100};  // [MeV]
    result.primaries_per_event = 1;
    result.num_events = 4;  // Overridden with BeamOn
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Enable optical tracking.
 */
auto OpticalSurfaces::make_setup_options() -> SetupOptions
{
    auto result = TMITestBase::make_setup_options();

    result.sd.enabled = false;
    result.optical = [] {
        OpticalSetupOptions opt;
        opt.capacity.tracks = 32768;
        opt.capacity.generators = 32768 * 8;
        opt.capacity.primaries = opt.capacity.generators;
        return opt;
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
void OpticalSurfaces::EndOfRunAction(G4Run const* run)
{
    auto& integration = detail::IntegrationSingleton::instance();
    if (integration.mode() == OffloadMode::enabled)
    {
        auto& local_transporter = integration.local_transporter();
        auto const& shared_params = integration.shared_params();

        // Check that local/shared data is available before end of run
        EXPECT_EQ(is_running_events(), static_cast<bool>(local_transporter));
        EXPECT_TRUE(shared_params) << "Celeritas was not enabled";

        auto const& optical_collector
            = shared_params.problem_loaded().optical_collector;
        EXPECT_TRUE(optical_collector) << "optical offloading was not enabled";
        if (local_transporter && optical_collector)
        {
            // Use diagnostic methods to check counters
            auto const& accum_stats
                = optical_collector->optical_state(local_transporter.GetState())
                      .accum();
            CELER_LOG_LOCAL(info)
                << "Ran " << accum_stats.steps << " over "
                << accum_stats.step_iters << " step iterations from "
                << accum_stats.flushes << " flushes";
            EXPECT_GT(accum_stats.steps, 0);
            EXPECT_GT(accum_stats.step_iters, 0);
            EXPECT_GT(accum_stats.flushes, 0);

            auto& aux_state = local_transporter.GetState().aux();
            auto counts = optical_collector->buffer_counts(aux_state);
            EXPECT_EQ(0, counts.buffer_size);  //!< Pending generators
            EXPECT_EQ(0, counts.num_pending);  //!< Photons pending generation
            EXPECT_EQ(0, counts.num_generated);  //!< Photons generated
        }
    }

    // Continue cleanup and other checks at end of run
    TMITestBase::EndOfRunAction(run);
}

//---------------------------------------------------------------------------//
/*!
 * Check that the test runs.
 */
TEST_F(OpticalSurfaces, run)
{
    auto& rm = this->run_manager();
    TMI::Instance().SetOptions(this->make_setup_options());

    CELER_LOG(status) << "Run initialization";
    rm.Initialize();
    CELER_LOG(status) << "Run two events";
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
