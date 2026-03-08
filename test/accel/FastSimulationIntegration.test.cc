//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/FastSimulationIntegration.test.cc
//---------------------------------------------------------------------------//
#include "accel/FastSimulationIntegration.hh"

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <regex>
#include <string_view>
#include <G4FastSimulationPhysics.hh>
#include <G4Region.hh>
#include <G4RegionStore.hh>
#include <G4RunManager.hh>
#include <G4UImanager.hh>
#include <G4VModularPhysicsList.hh>

#include "corecel/StringSimplifier.hh"
#include "corecel/cont/Array.hh"
#include "corecel/io/Logger.hh"
#include "geocel/GeantUtils.hh"
#include "geocel/ScopedGeantExceptionHandler.hh"
#include "geocel/UnitUtils.hh"
#include "celeritas/ext/GeantParticleView.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/inp/Events.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/OpticalCollector.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "accel/FastSimulationModel.hh"
#include "accel/LocalTransporter.hh"
#include "accel/SetupOptions.hh"
#include "accel/SharedParams.hh"
#include "accel/detail/IntegrationSingleton.hh"

#include "IntegrationTestBase.hh"
#include "celeritas_test.hh"

using FSI = celeritas::FastSimulationIntegration;

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

}  // namespace

//---------------------------------------------------------------------------//
// TEST BASE
//---------------------------------------------------------------------------//
/*!
 * Test the FastSimulationIntegration (FSI).
 *
 * The fast simulation will:
 * - Add a physics constructor that sets up fast simulation for the supported
 *   particles
 * - Add the fast simulation model to the default region
 *   - TODO: Region name(s) from SetupOptions
 *   - TODO: Offload particles from SetupOptions
 * - Set up Celeritas shared data at BeginOfRunAction on the main thread
 * - Set up Celeritas local data at BeginOfRunAction on the worker thread
 * - Clean up on EndOfRunAction
 */
class FSITestBase : virtual public IntegrationTestBase
{
    using Base = IntegrationTestBase;

  protected:
    // Physics
    UPPhysicsList make_physics_list() const override
    {
        auto physics = Base::make_physics_list();
        CELER_ASSERT(physics);
        auto fast_physics = new G4FastSimulationPhysics();
        fast_physics->ActivateFastSimulation("e-");
        fast_physics->ActivateFastSimulation("e+");
        fast_physics->ActivateFastSimulation("gamma");
        physics->RegisterPhysics(fast_physics);
        return physics;
    }
    void ConstructSDandField() override
    {
        G4Region* region = G4RegionStore::GetInstance()->GetRegion(
            "DefaultRegionForTheWorld");
        ASSERT_NE(region, nullptr);
        // Underlying GVFastSimulationModel constructor handles ownership,
        // so we must ignore the returned pointer...
        new FastSimulationModel(region);
    }
    void BeginOfRunAction(G4Run const* run) override
    {
        FSI::Instance().BeginOfRunAction(run);
    }
    void EndOfRunAction(G4Run const* run) override
    {
        FSI::Instance().EndOfRunAction(run);
    }
    void BeginOfEventAction(G4Event const*) override {}
    void EndOfEventAction(G4Event const*) override {}

    std::function<void()> check_during_run_;
};

//---------------------------------------------------------------------------//
// LAR SPHERE (EM only)
//---------------------------------------------------------------------------//
class LarSphere : public LarSphereIntegrationMixin, public FSITestBase
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
            return FSITestBase::caught_g4_runtime_error(e);
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
    FSI::Instance().SetOptions(this->make_setup_options());

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
    auto& fsi = FSI::Instance();

    EXPECT_EQ(fsi.GetMode(), OffloadMode::uninitialized);
    fsi.SetOptions(this->make_setup_options());
    EXPECT_NE(fsi.GetMode(), OffloadMode::uninitialized);

    std::atomic<int> check_count{0};

    auto& ui = *G4UImanager::GetUIpointer();
    if (SharedParams::GetMode() != OffloadMode::disabled)
    {
        ui.ApplyCommand("/celer/maxNumTracks 128");
        ui.ApplyCommand("/celer/maxInitializers 10000");

        check_during_run_ = [&check_count, &fsi] {
            EXPECT_NE(OffloadMode::uninitialized, fsi.GetMode());

            if (fsi.GetMode() == OffloadMode::enabled && is_running_events())
            {
                CELER_LOG_LOCAL(debug) << "Checking number of tracks";
                ++check_count;

                auto const& state = fsi.GetState();
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
// TESTEM3
//---------------------------------------------------------------------------//
class TestEm3 : public TestEm3IntegrationMixin, public FSITestBase
{
};

/*!
 * Check that TestEm3 runs.
 */
TEST_F(TestEm3, run)
{
    auto& rm = this->run_manager();
    FSI::Instance().SetOptions(this->make_setup_options());

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
// OP-NOVICE OPTICAL
//---------------------------------------------------------------------------//
class OpNoviceOptical : public OpNoviceIntegrationMixin, public FSITestBase
{
};

TEST_F(OpNoviceOptical, run)
{
    auto& rm = this->run_manager();
    FSI::Instance().SetOptions(this->make_setup_options());

    rm.Initialize();
    rm.BeamOn(2);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
