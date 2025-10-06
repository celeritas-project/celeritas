//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/ScintOffloadExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/global/CoreTrackView.hh"

#include "../OffloadData.hh"
#include "../ScintillationOffload.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// LAUNCHER
//---------------------------------------------------------------------------//
/*!
 * Generate optical distribution data.
 *
 * Note that the track may be inactive! TODO: we could add a `user_start`
 * action to clear distribution data rather than applying it to inactive tracks
 * at every step.
 */
struct ScintOffloadExecutor
{
    inline CELER_FUNCTION void
    operator()(celeritas::CoreTrackView const& track);

    NativeCRef<celeritas::optical::MaterialParamsData> const material;
    NativeCRef<ScintillationData> const scint;
    NativeRef<optical::GeneratorStateData> const offload;
    NativeRef<OffloadStepStateData> const steps;
    size_type buffer_size;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Generate optical distribution data.
 */
CELER_FUNCTION void ScintOffloadExecutor::operator()(CoreTrackView const& track)
{
    CELER_EXPECT(scint);
    CELER_EXPECT(offload);
    CELER_EXPECT(steps);

    using DistId = ItemId<optical::GeneratorDistributionData>;

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

    Real3 const& pos = track.geometry().pos();
    auto edep = track.physics_step().energy_deposition();
    auto particle = track.particle();
    auto rng = track.rng();

    // Get the distribution data used to generate scintillation optical photons
    dist = ScintillationOffload(particle, sim, pos, edep, scint, step)(rng);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
