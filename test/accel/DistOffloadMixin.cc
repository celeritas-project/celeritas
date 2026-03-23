//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/DistOffloadMixin.cc
//---------------------------------------------------------------------------//
#include "DistOffloadMixin.hh"

#include <memory>
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
 * Process a G4 step: count particles and offload optical distributions.
 */
void DistOffloadCounter::operator()(StreamId stream, G4Step const& step)
{
    // Count all tracks by type
    GeantParticleView pv{*step.GetTrack()->GetParticleDefinition()};
    auto& ctrs = counters_[stream.get()];
    if (pv.is_optical_photon())
    {
        ++ctrs.optical;
    }
    else
    {
        ++ctrs.other;
    }

    if (IntegrationTestBase::test_offload() == TestOffload::g4)
    {
        return;
    }

    if (step.GetStepLength() == 0)
    {
        // Skip "no-process"-defined steps
        return;
    }

    auto* pm = step.GetTrack()->GetParticleDefinition()->GetProcessManager();
    CELER_ASSERT(pm);

    // Determine how many Cherenkov and scintillation photons to generate
    size_type num_cherenkov{0};
    size_type num_scintillation{0};
    if (auto const* p = find_process<G4Cerenkov>(pm, "Cerenkov"))
    {
        num_cherenkov = p->GetNumPhotons();
    }
    if (auto const* p = find_process<G4Scintillation>(pm, "Scintillation"))
    {
        num_scintillation = p->GetNumPhotons();
    }

    if (num_cherenkov == 0 && num_scintillation == 0)
    {
        return;
    }

    std::call_once(geant_geo_once_, [this] {
        geant_geo_ = celeritas::global_geant_geo().lock();
        CELER_VALIDATE(geant_geo_, << "global Geant4 geometry is not loaded");
    });

    auto* pre_step = step.GetPreStepPoint();
    auto* post_step = step.GetPostStepPoint();
    CELER_ASSERT(pre_step && post_step);

    // Create distribution and push to Celeritas
    optical::GeneratorDistributionData data;
    data.step_length
        = native_from_geant<lengthunits::ClhepLength>(step.GetStepLength());
    data.charge = units::ElementaryCharge{
        static_cast<real_type>(post_step->GetCharge())};
    auto& pre = data.points[StepPoint::pre];
    pre.speed = units::LightSpeed(pre_step->GetBeta());
    pre.time = native_from_geant<units::ClhepTime>(pre_step->GetGlobalTime());
    pre.pos = native_from_geant<lengthunits::ClhepLength, real_type>(
        pre_step->GetPosition());
    auto& post = data.points[StepPoint::post];
    post.speed = units::LightSpeed(post_step->GetBeta());
    post.time = native_from_geant<units::ClhepTime>(post_step->GetGlobalTime());
    post.pos = native_from_geant<lengthunits::ClhepLength, real_type>(
        post_step->GetPosition());
    auto* g4mat = pre_step->GetMaterial();
    CELER_ASSERT(g4mat);
    data.material
        = (*geant_geo_->geo_optical_id_map())[geant_geo_->geant_to_id(*g4mat)];

    auto& local = detail::IntegrationSingleton::instance().local_offload();
    auto& gen_offload = dynamic_cast<LocalOpticalGenOffload&>(local);
    if (num_cherenkov > 0)
    {
        data.type = GeneratorType::cherenkov;
        data.num_photons = num_cherenkov;
        CELER_ASSERT(data);
        gen_offload.Push(data);
    }
    if (num_scintillation > 0)
    {
        data.type = GeneratorType::scintillation;
        data.num_photons = num_scintillation;
        CELER_ASSERT(data);
        gen_offload.Push(data);
    }
    CELER_LOG(debug) << "Generating " << num_cherenkov
                     << " Cherenkov photons and " << num_scintillation
                     << " scintillation photons";
}

//---------------------------------------------------------------------------//
/*!
 * Sum all per-stream counters.
 *
 * This is not thread safe: call only from the master thread at end of run.
 */
StepCounters DistOffloadCounter::merged() const
{
    StepCounters result;
    for (auto const& c : counters_)
    {
        result.optical += c.optical;
        result.other += c.other;
    }
    return result;
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
        optical->scintillation->stack_photons = false;
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
auto DistOffloadMixin::make_setup_options() const -> SetupOptions
{
    auto result = IntegrationTestBase::make_setup_options();

    result.optical = [] {
        OpticalSetupOptions opt;
        opt.capacity.tracks = 32768;
        opt.capacity.generators = opt.capacity.tracks * 8;
        opt.capacity.primaries = opt.capacity.tracks * 16;

        // Enable optical distribution offloading
        opt.generator = inp::OpticalOffloadGenerator{};

        return opt;
    }();

    // Don't offload any particles
    result.offload_particles = SetupOptions::VecG4PD{};

    return result;
}

//---------------------------------------------------------------------------//
auto DistOffloadMixin::make_step_callback() -> StepCallback
{
    counter_.emplace(this->num_streams());
    auto* ctr = &*counter_;
    return [ctr](StreamId stream, G4Step const& step) { (*ctr)(stream, step); };
}

//---------------------------------------------------------------------------//
/*!
 * Check counters at end-of-run on master.
 */
void DistOffloadMixin::EndOfRunAction(G4Run const*)
{
    if (G4Threading::IsMasterThread())
    {
        auto counters = counter_->merged();
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

}  // namespace test
}  // namespace celeritas
