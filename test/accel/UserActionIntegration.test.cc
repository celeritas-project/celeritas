//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/UserActionIntegration.test.cc
//---------------------------------------------------------------------------//
#include "accel/UserActionIntegration.hh"

#include <memory>
#include <G4Cerenkov.hh>
#include <G4ProcessManager.hh>
#include <G4RunManager.hh>
#include <G4Scintillation.hh>
#include <G4Step.hh>
#include <G4StepPoint.hh>

#include "geocel/ScopedGeantExceptionHandler.hh"
#include "geocel/UnitUtils.hh"
#include "geocel/g4/Convert.hh"
#include "accel/SetupOptions.hh"
#include "accel/detail/IntegrationSingleton.hh"

#include "IntegrationTestBase.hh"
#include "celeritas_test.hh"

using UAI = celeritas::UserActionIntegration;

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
class UAITrackingAction final : public G4UserTrackingAction
{
    void PreUserTrackingAction(G4Track const* track) final
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
    rm.BeamOn(1);
    cout << "second run done" << endl;
}

//---------------------------------------------------------------------------//
// LAR SPHERE WITH OPTICAL OFFLOAD
//---------------------------------------------------------------------------//
class LSOOSteppingAction final : public G4UserSteppingAction
{
    void UserSteppingAction(G4Step const* step) final;
};

//---------------------------------------------------------------------------//
/*!
 * Offload optical distributions.
 */
class LarSphereOpticalOffload : public LarSphere
{
  public:
    PrimaryInput make_primary_input() const override;
    PhysicsInput make_physics_input() const override;
    SetupOptions make_setup_options() override;
    UPStepAction make_stepping_action() override
    {
        return std::make_unique<LSOOSteppingAction>();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Single electron primary.
 */
auto LarSphereOpticalOffload::make_primary_input() const -> PrimaryInput
{
    using MevEnergy = Quantity<units::Mev, double>;
    auto result = LarSphereIntegrationMixin::make_primary_input();

    result.shape = inp::PointDistribution{from_cm({0.1, 0.1, 0})};
    result.primaries_per_event = 1;
    result.energy = inp::MonoenergeticDistribution{MevEnergy{1}};
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Enable optical physics and disable photon stacking.
 */
auto LarSphereOpticalOffload::make_physics_input() const -> PhysicsInput
{
    auto result = LarSphereIntegrationMixin::make_physics_input();

    // Set default optical physics
    auto& optical = result.optical;
    optical = {};

    // Disable generation of Cherenkov and scintillation photons in Geant4
    optical.cherenkov.stack_photons = false;
    optical.scintillation.stack_photons = false;

    // Disable WLS which isn't yet working (reemission) in Celeritas
    using WLSO = WavelengthShiftingOptions;
    optical.wavelength_shifting = WLSO::deactivated();
    optical.wavelength_shifting2 = WLSO::deactivated();

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Enable optical tracking with distribution offloading.
 */
auto LarSphereOpticalOffload::make_setup_options() -> SetupOptions
{
    auto result = LarSphereIntegrationMixin::make_setup_options();

    result.optical_capacity = [] {
        inp::OpticalStateCapacity cap;
        cap.tracks = 32768;
        cap.generators = cap.tracks * 8;
        cap.primaries = cap.tracks * 16;
        return cap;
    }();

    // Enable optical distribution offloading
    result.optical_generator = inp::OpticalOffloadGenerator{};

    // Don't offload any particles
    result.offload_particles = SetupOptions::VecG4PD{};

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Stepping action for pushing optical distributions to Celeritas.
 */
void LSOOSteppingAction::UserSteppingAction(G4Step const* step)
{
    CELER_EXPECT(step);

    constexpr double clhep_time{1 / units::nanosecond};

    auto& local = detail::IntegrationSingleton::local_optical_offload();
    if (!local)
    {
        // Offloading is disabled
        return;
    }

    if (step->GetStepLength() == 0)
    {
        // Skip "no-process"-defined steps
        return;
    }

    auto* pm = step->GetTrack()->GetDefinition()->GetProcessManager();
    CELER_ASSERT(pm);

    // Determine how many Cherenkov and scintillation photons to generate
    size_type num_cherenkov{0};
    size_type num_scintillation{0};
    if (auto const* p = dynamic_cast<G4Cerenkov const*>(pm->GetProcess("Cerenk"
                                                                       "ov")))
    {
        num_cherenkov = p->GetNumPhotons();
    }
    if (auto const* p = dynamic_cast<G4Scintillation const*>(
            pm->GetProcess("Scintillation")))
    {
        num_scintillation = p->GetNumPhotons();
    }

    if (num_cherenkov == 0 && num_scintillation == 0)
    {
        return;
    }

    auto* pre_step = step->GetPreStepPoint();
    auto* post_step = step->GetPostStepPoint();
    CELER_ASSERT(pre_step && post_step);

    // Create distribution and push to Celeritas
    // TODO: Get optical material ID
    // TODO: Does the post-step speed account for only continuous energy
    // loss or continuous+discrete?
    optical::GeneratorDistributionData data;
    data.time = convert_from_geant(post_step->GetGlobalTime(), clhep_time);
    data.step_length = convert_from_geant(step->GetStepLength(), clhep_length);
    data.charge = units::ElementaryCharge{
        static_cast<real_type>(post_step->GetCharge())};
    data.material = OptMatId(0);
    data.points[StepPoint::pre]
        = {units::LightSpeed(pre_step->GetBeta()),
           convert_from_geant(pre_step->GetPosition(), clhep_length)};
    data.points[StepPoint::post]
        = {units::LightSpeed(post_step->GetBeta()),
           convert_from_geant(post_step->GetPosition(), clhep_length)};

    if (num_cherenkov > 0)
    {
        data.type = GeneratorType::cherenkov;
        data.num_photons = num_cherenkov;
        CELER_ASSERT(data);
        local.Push(data);
    }
    if (num_scintillation > 0)
    {
        data.type = GeneratorType::scintillation;
        data.num_photons = num_scintillation;
        CELER_ASSERT(data);
        local.Push(data);
    }
    CELER_LOG(debug) << "Generating " << num_cherenkov
                     << " Cherenkov photons and " << num_scintillation
                     << " scintillation photons";
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
}  // namespace test
}  // namespace celeritas
