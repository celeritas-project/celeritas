//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/detail/ActionDiagnosticImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/math/Atomics.hh"
#include "celeritas/global/CoreTrackView.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Tally post-step actions by particle type.
 */
inline CELER_FUNCTION void
tally_action(CoreTrackView const& track,
             NativeRef<ActionDiagnosticStateData>&& data)
{
    using BinId = ItemId<size_type>;

    CELER_EXPECT(data);

    ActionId aid = track.make_sim_view().step_limit().action;
    CELER_ASSERT(aid);
    ParticleId pid = track.make_particle_view().particle_id();
    CELER_ASSERT(pid);
    size_type num_particles = track.make_physics_view().num_particles();

    BinId bin{aid.unchecked_get() * num_particles + pid.unchecked_get()};
    CELER_ASSERT(bin < data.counts.size());
    celeritas::atomic_add(&data.counts[bin], size_type(1));
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
