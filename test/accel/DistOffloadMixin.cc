//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/DistOffloadMixin.cc
//---------------------------------------------------------------------------//
#include "DistOffloadMixin.hh"

#include <memory>
#include <mutex>
#include <G4Cerenkov.hh>
#include <G4ProcessManager.hh>
#include <G4Scintillation.hh>
#include <G4Step.hh>

#include "corecel/io/Logger.hh"
#include "geocel/GeoOpticalIdMap.hh"
#include "geocel/g4/Convert.hh"
#include "celeritas/ext/GeantParticleView.hh"
#include "celeritas/optical/gen/GeneratorData.hh"
#include "accel/IntegrationTestBase.hh"
#include "accel/LocalOpticalGenOffload.hh"
#include "accel/detail/IntegrationSingleton.hh"
#include "accel/gen/CherenkovOffload.hh"
#include "accel/gen/ScintillationOffload.hh"

namespace celeritas
{
namespace test
{
namespace
{
//---------------------------------------------------------------------------//

template<class T>
T const* find_process(G4ProcessManager* pm, std::string const& name)
{
    return dynamic_cast<T const*>(pm->GetProcess(name));
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Stepping action for pushing optical distributions to Celeritas.
 */
void DistOffloadSteppingAction::UserSteppingAction(G4Step const* step)
{
    CELER_EXPECT(step);

    // Geant4
    GeantParticleView pv{*step->GetTrack()->GetParticleDefinition()};
    if (pv.is_optical_photon())
    {
        ++counters_->optical;
    }
    else
    {
        ++counters_->other;
    }
}

//---------------------------------------------------------------------------//
// DistOffloadMixin
//---------------------------------------------------------------------------//
/*!
 * Enable optical physics and disable photon stacking.
 */
auto DistOffloadMixin::make_physics_input() const -> PhysicsInput
{
    auto result = IntegrationTestBase::make_physics_input();

    // Set default optical physics
    auto& optical = result.optical;
    optical.emplace();

    if (IntegrationTestBase::test_offload() != TestOffload::g4)
    {
        // Disable generation of Cherenkov and scintillation photons in Geant4,
        // since we're killing or sending to Celeritas
        optical->cherenkov->stack_photons = false;
        optical->cherenkov->custom_cherenkov
            = []() { return std::make_unique<CherenkovOffload>(); };
        optical->scintillation->stack_photons = false;
        optical->scintillation->custom_scintillation
            = []() { return std::make_unique<ScintillationOffload>(); };
    }

    // Disable WLS which isn't yet working (reemission) in Celeritas
    optical->wavelength_shifting = std::nullopt;
    optical->wavelength_shifting2 = std::nullopt;

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Enable optical tracking with distribution offloading.
 */
auto DistOffloadMixin::make_setup_options() -> SetupOptions
{
    auto result = IntegrationTestBase::make_setup_options();

    result.optical = [] {
        OpticalSetupOptions opt;
        opt.capacity.tracks = 32768;
        opt.capacity.generators = *opt.capacity.tracks * 8;
        opt.capacity.primaries = *opt.capacity.tracks * 16;

        // Enable optical distribution offloading
        opt.generator = inp::OpticalOffloadGenerator{};

        return opt;
    }();

    // Don't offload any particles
    result.offload_particles = SetupOptions::VecG4PD{};

    return result;
}

//---------------------------------------------------------------------------//
auto DistOffloadMixin::make_stepping_action() -> UPStepAction
{
    static std::mutex mu_;
    std::lock_guard scoped_lock{mu_};
    auto local_ctrs = std::make_shared<StepCounters>();
    counters_.push_back(local_ctrs);
    return std::make_unique<DistOffloadSteppingAction>(std::move(local_ctrs));
}

//---------------------------------------------------------------------------//
/*!
 * Check counters at end-of-run on master.
 */
void DistOffloadMixin::EndOfRunAction(G4Run const*)
{
    if (G4Threading::IsMasterThread())
    {
        auto counters = this->merge_step_counters();
        EXPECT_NE(counters.other, 0);
        if (IntegrationTestBase::test_offload() != TestOffload::g4)
        {
            // No optical photons should've been stacked or stepped in G4
            EXPECT_EQ(0, counters.optical);
        }
        else
        {
            // Geant4 should have run some optical photons
            EXPECT_NE(0, counters.optical);
        }
        CELER_LOG(info) << "Total Geant4 steps: " << counters.optical
                        << " optical, " << counters.other << " other";
    }
}

//---------------------------------------------------------------------------//
/*!
 * Sum counters across all threads.
 *
 * This is not thread safe: do it only in the master end of run.
 */
StepCounters DistOffloadMixin::merge_step_counters() const
{
    StepCounters result;
    for (auto const& c : counters_)
    {
        result.optical += c->optical;
        result.other += c->other;
    }
    return result;
}

}  // namespace test
}  // namespace celeritas
