//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/CherenkovOffloadExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/global/CoreTrackView.hh"

#include "../CherenkovOffload.hh"
#include "../OffloadData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// LAUNCHER
//---------------------------------------------------------------------------//
/*!
 * Generate optical distribution data.
 */
struct CherenkovOffloadExecutor
{
    inline CELER_FUNCTION void
    operator()(celeritas::CoreTrackView const& track);

    NativeCRef<celeritas::optical::MaterialParamsData> material;
    NativeCRef<CherenkovData> cherenkov;
    NativeRef<GeneratorStateData> offload;
    NativeRef<OffloadStepStateData> steps;
    size_type buffer_size;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Generate optical distribution data.
 */
CELER_FUNCTION void
CherenkovOffloadExecutor::operator()(CoreTrackView const& track)
{
    CELER_EXPECT(material);
    CELER_EXPECT(cherenkov);
    CELER_EXPECT(offload);
    CELER_EXPECT(steps);

    using DistId = ItemId<GeneratorDistributionData>;

    auto tsid = track.track_slot_id();
    CELER_ASSERT(buffer_size + tsid.get() < offload.distributions.size());
    auto& dist = offload.distributions[DistId(buffer_size + tsid.get())];

    // Clear distribution data
    dist = {};

    auto sim = track.sim();
    auto const& step = steps.step[tsid];

    if (!step || sim.status() == TrackStatus::inactive)
    {
        // Inactive tracks, materials with no optical properties, or particles
        // that started the step with zero energy (e.g. a stopped positron)
        return;
    }

    auto particle = track.particle();

    // Get the distribution data used to generate Cherenkov optical photons
    if (particle.charge() != zero_quantity())
    {
        CherenkovOffload sample_dist(
            // Pre-step:
            step,
            optical::MaterialView{material, step.material},
            // Post-step:
            particle,
            sim,
            track.geometry().pos(),
            cherenkov);
        auto rng = track.rng();
        dist = sample_dist(rng);
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
