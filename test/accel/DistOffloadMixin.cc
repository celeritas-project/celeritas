//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/DistOffloadMixin.cc
//---------------------------------------------------------------------------//
#include "DistOffloadMixin.hh"

#include <G4Cerenkov.hh>
#include <G4ProcessManager.hh>
#include <G4Scintillation.hh>
#include <G4Step.hh>

#include "corecel/io/Logger.hh"
#include "geocel/GeantGeoParams.hh"
#include "geocel/g4/Convert.hh"
#include "celeritas/optical/gen/GeneratorData.hh"
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
 * Stepping action for pushing optical distributions to Celeritas.
 */
void DistOffloadSteppingAction::UserSteppingAction(G4Step const* step)
{
    CELER_EXPECT(step);

    constexpr double clhep_time{1 / units::nanosecond};

    auto& local = detail::IntegrationSingleton::instance().local_offload();
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

    auto* pre_step = step->GetPreStepPoint();
    auto* post_step = step->GetPostStepPoint();
    CELER_ASSERT(pre_step && post_step);

    // Create distribution and push to Celeritas
    // TODO: Does the post-step speed account for only continuous energy
    // loss or continuous+discrete?
    optical::GeneratorDistributionData data;
    data.time = convert_from_geant(post_step->GetGlobalTime(), clhep_time);
    data.step_length = convert_from_geant(step->GetStepLength(), clhep_length);
    data.charge = units::ElementaryCharge{
        static_cast<real_type>(post_step->GetCharge())};

    // Get geant4 geometry wrapper as a translation layer
    if (CELER_UNLIKELY(!geant_geo_))
    {
        geant_geo_ = global_geant_geo().lock();
    }
    CELER_ASSERT(geant_geo_);

    auto* mat = pre_step->GetMaterial();
    CELER_ASSERT(mat);
    GeoMatId gm = geant_geo_->geant_to_id(*mat);
    // TODO: map geo -> phys -> optical matids?!
    // Or logical volume -> optical matids?
    CELER_DISCARD(gm);

    data.material = OptMatId(0);
    data.points[StepPoint::pre]
        = {units::LightSpeed(pre_step->GetBeta()),
           convert_from_geant(pre_step->GetPosition(), clhep_length)};
    data.points[StepPoint::post]
        = {units::LightSpeed(post_step->GetBeta()),
           convert_from_geant(post_step->GetPosition(), clhep_length)};

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
    optical = {};

    // TODO: this should *not* be disabled if we're running the test in G4-only
    // mode
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
auto DistOffloadMixin::make_setup_options() -> SetupOptions
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
}  // namespace test
}  // namespace celeritas
