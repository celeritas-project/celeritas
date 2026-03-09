//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/UserActionIntegration.test.cc
//---------------------------------------------------------------------------//
#include "accel/UserActionIntegration.hh"

#include <memory>
#include <G4OpticalPhoton.hh>
#include <G4RunManager.hh>
#include <G4Step.hh>
#include <G4StepPoint.hh>

#include "corecel/math/ArrayUtils.hh"
#include "geocel/ScopedGeantExceptionHandler.hh"
#include "geocel/UnitUtils.hh"
#include "accel/LocalOpticalGenOffload.hh"
#include "accel/SetupOptions.hh"
#include "accel/detail/IntegrationSingleton.hh"

#include "DistOffloadMixin.hh"
#include "IntegrationTestBase.hh"
#include "celeritas_test.hh"

using UAI = celeritas::UserActionIntegration;

namespace celeritas
{
namespace test
{

constexpr bool using_surface_vg = CELERITAS_VECGEOM_SURFACE
                                  && CELERITAS_CORE_GEO
                                         == CELERITAS_CORE_GEO_VECGEOM;

//---------------------------------------------------------------------------//
class UAITrackingAction : public G4UserTrackingAction
{
  public:
    void PreUserTrackingAction(G4Track const* track) override
    {
        UAI::Instance().PreUserTrackingAction(const_cast<G4Track*>(track));
    }
};

//---------------------------------------------------------------------------//
class UAITestBase : virtual public IntegrationTestBase
{
    using Base = IntegrationTestBase;

  protected:
    void BeginOfRunAction(G4Run const* run) override
    {
        UAI::Instance().BeginOfRunAction(run);
    }
    void EndOfRunAction(G4Run const* run) override
    {
        UAI::Instance().EndOfRunAction(run);
    }
    void BeginOfEventAction(G4Event const* event) override
    {
        UAI::Instance().BeginOfEventAction(event);
    }
    void EndOfEventAction(G4Event const* event) override
    {
        UAI::Instance().EndOfEventAction(event);

        auto& local = detail::IntegrationSingleton::instance().local_offload();
        if (!local)
            return;

        EXPECT_EQ(0, local.GetBufferSize());
    }
    UPTrackAction make_tracking_action() override
    {
        return std::make_unique<UAITrackingAction>();
    }
};

//---------------------------------------------------------------------------//
// LAR SPHERE
//---------------------------------------------------------------------------//
class LarSphere : public LarSphereIntegrationMixin, public UAITestBase
{
};

TEST_F(LarSphere, run)
{
    auto& rm = this->run_manager();
    UAI::Instance().SetOptions(this->make_setup_options());

    cout << "initializing" << endl;
    rm.Initialize();
    cout << "beam on" << endl;

    rm.BeamOn(3);
    cout << "initial run done" << endl;

    if (using_surface_vg)
    {
        GTEST_SKIP() << "VecGeom surface model does not support multiple runs";
    }

    rm.BeamOn(1);
    cout << "second run done" << endl;
}

//---------------------------------------------------------------------------//
// LAR SPHERE WITH OPTICAL OFFLOAD
//---------------------------------------------------------------------------//
/*!
 * Offload optical distributions.
 *
 * The \c LarSphere base sets up geometry and primaries (electrons), and \c
 * DistOffloadMixin sets up the correct Geant4 physics and Celeritas run
 * options.
 */
class LarSphereOpticalOffload : public DistOffloadMixin, public LarSphere
{
  public:
    PrimaryInput make_primary_input() const override;

    void EndOfRunAction(G4Run const* run) override
    {
        DistOffloadMixin::EndOfRunAction(run);
        LarSphere::EndOfRunAction(run);
    }
};

//---------------------------------------------------------------------------//
/*!
 * Single electron primary.
 */
auto LarSphereOpticalOffload::make_primary_input() const -> PrimaryInput
{
    auto result = LarSphere::make_primary_input();

    result.shape
        = inp::PointDistribution{array_cast<double>(from_cm({0.1, 0.1, 0}))};
    result.primaries_per_event = 10;
    result.energy = inp::MonoenergeticDistribution{1};  // [MeV]
    return result;
}

//---------------------------------------------------------------------------//
TEST_F(LarSphereOpticalOffload, run)
{
    auto& rm = this->run_manager();
    UAI::Instance().SetOptions(this->make_setup_options());

    rm.Initialize();
    rm.BeamOn(2);
}

//---------------------------------------------------------------------------//
// TEST EM3
//---------------------------------------------------------------------------//
class TestEm3 : public TestEm3IntegrationMixin, public UAITestBase
{
};

TEST_F(TestEm3, run)
{
    // Test loading instance before run manager
    auto& uai = UAI::Instance();

    {
        // Options can't be set before run manager is initialized
        ScopedGeantExceptionHandler convert_to_throw;
        EXPECT_THROW(uai.SetOptions(SetupOptions{}), RuntimeError);
    }

    auto& rm = this->run_manager();
    // Set options for real
    uai.SetOptions(this->make_setup_options());

    rm.Initialize();
    rm.BeamOn(2);
}

//---------------------------------------------------------------------------//
// OP-NOVICE OPTICAL
//---------------------------------------------------------------------------//
class OpNoviceOptical : public OpNoviceIntegrationMixin, public UAITestBase
{
};

TEST_F(OpNoviceOptical, run)
{
    auto& rm = this->run_manager();
    UAI::Instance().SetOptions(this->make_setup_options());

    rm.Initialize();
    rm.BeamOn(2);
}

//---------------------------------------------------------------------------//
// LAR SPHERE WITH OPTICAL TRACK OFFLOAD
//---------------------------------------------------------------------------//
class LSOOTrackingAction final : public UAITrackingAction
{
  public:
    void PreUserTrackingAction(G4Track const* track) final
    {
        UAITrackingAction::PreUserTrackingAction(track);

        auto& local = detail::IntegrationSingleton::instance().local_offload();
        auto* opt_offload = dynamic_cast<LocalOpticalTrackOffload*>(&local);

        if (opt_offload && opt_offload->Initialized())
        {
            pushes_ = opt_offload->num_pushed();
        }
    }
    std::size_t num_pushes() const { return pushes_; }

  private:
    std::size_t pushes_{0};
};

//---------------------------------------------------------------------------//
/*!
 * Offload optical tracks.
 */
class LarSphereOpticalTrackOffload : public LarSphere
{
  public:
    PhysicsInput make_physics_input() const override;
    PrimaryInput make_primary_input() const override;
    SetupOptions make_setup_options() override;
    void EndOfRunAction(G4Run const* run) override;
    UPTrackAction make_tracking_action() override
    {
        auto result = std::make_unique<LSOOTrackingAction>();
        {
            static std::mutex mutex;
            std::lock_guard<std::mutex> lock(mutex);
            tracking_.push_back(result.get());
        }
        return result;
    }

  private:
    std::vector<LSOOTrackingAction*> tracking_;
};

//---------------------------------------------------------------------------//
/*!
 * Single electron primary.
 */
auto LarSphereOpticalTrackOffload::make_primary_input() const -> PrimaryInput
{
    auto result = LarSphere::make_primary_input();
    result.shape
        = inp::PointDistribution{array_cast<double>(from_cm({0.1, 0.1, 0}))};
    result.primaries_per_event = 1;
    result.energy = inp::MonoenergeticDistribution{1};
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Enable optical physics
 */
auto LarSphereOpticalTrackOffload::make_physics_input() const -> PhysicsInput
{
    auto result = LarSphere::make_physics_input();

    // Set default optical physics
    auto& optical = result.optical;
    optical.emplace();
    optical->cherenkov->stack_photons = true;
    optical->scintillation->stack_photons = true;

    optical->wavelength_shifting = std::nullopt;
    optical->wavelength_shifting2 = std::nullopt;

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Enable optical tracking offloading.
 */
auto LarSphereOpticalTrackOffload::make_setup_options() -> SetupOptions
{
    auto result = LarSphereIntegrationMixin::make_setup_options();
    result.optical = [] {
        OpticalSetupOptions opt;
        opt.capacity.tracks = 32;
        opt.capacity.generators = opt.capacity.tracks * 8;
        opt.capacity.primaries = opt.capacity.tracks * 16;
        opt.generator = inp::OpticalDirectGenerator{};
        return opt;
    }();

    // Offload optical photon
    result.offload_particles
        = SetupOptions::VecG4PD{G4OpticalPhoton::Definition()};

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Test that the optical track offload was successful.
 */
void LarSphereOpticalTrackOffload::EndOfRunAction(G4Run const* run)
{
    auto& integration = detail::IntegrationSingleton::instance();
    auto& local = integration.local_offload();

    auto test_mode = IntegrationTestBase::test_offload();

    if (!G4Threading::IsMultithreadedApplication())
    {
        if (test_mode == TestOffload::cpu || test_mode == TestOffload::gpu)
        {
            auto* loto = dynamic_cast<LocalOpticalTrackOffload*>(&local);
            CELER_VALIDATE(loto, << "not a local optical track offload");

            // Validate that we intercepted optical tracks
            EXPECT_GT(loto->num_pushed(), 0)
                << R"(should have pushed many optical tracks)";
        }
    }
    if (G4Threading::IsMultithreadedApplication())
    {
        std::size_t total_tracks_pushed{0};
        for (auto* action : tracking_)
        {
            total_tracks_pushed += action->num_pushes();
        }

        CELER_LOG(info) << "Celeritas offloaded  " << total_tracks_pushed
                        << " optical tracks";

        if (test_mode == TestOffload::g4 || test_mode == TestOffload::ko)
        {
            EXPECT_EQ(total_tracks_pushed, 0);
        }
    }
    // Continue cleanup and other checks at end of run
    LarSphere::EndOfRunAction(run);
}

//---------------------------------------------------------------------------//
TEST_F(LarSphereOpticalTrackOffload, run)
{
    auto& rm = this->run_manager();
    UAI::Instance().SetOptions(this->make_setup_options());

    rm.Initialize();
    rm.BeamOn(1);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
