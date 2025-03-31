//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/CherenkovOffloadExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/optical/CherenkovOffload.hh"
#include "celeritas/optical/OffloadData.hh"

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

    NativeCRef<celeritas::optical::MaterialParamsData> const material;
    NativeCRef<celeritas::optical::CherenkovData> const cherenkov;
    NativeRef<OffloadStateData> const state;
    OpticalOffloadCounters<> size;
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
    CELER_EXPECT(state);
    CELER_EXPECT(cherenkov);
    CELER_EXPECT(material);

    using DistId = ItemId<celeritas::optical::GeneratorDistributionData>;

    auto tsid = track.track_slot_id();
    CELER_ASSERT(size.cherenkov + tsid.get() < state.cherenkov.size());
    auto& cherenkov_dist = state.cherenkov[DistId(size.cherenkov + tsid.get())];

    // Clear distribution data
    cherenkov_dist = {};

    auto sim = track.sim();
    auto const& step = state.step[tsid];

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
        Real3 const& pos = track.geometry().pos();
        optical::MaterialView opt_mat{material, step.material};
        auto rng = track.rng();

        CherenkovOffload generate(particle, sim, opt_mat, pos, cherenkov, step);
        cherenkov_dist = generate(rng);
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
