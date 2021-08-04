//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file InitializeTracks.cc
//---------------------------------------------------------------------------//
#include "InitializeTracks.hh"

#include "base/Range.hh"
#include "base/Types.hh"
#include "sim/TrackInitInterface.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Create track initializers on device from primary particles.
 */
void process_primaries(Span<const Primary>          primaries,
                       const TrackInitStateHostRef& inits)
{
    // TODO: What to do about celeritas::size_type for host?
    for (auto tid :
         range(ThreadId{static_cast<celeritas::size_type>(primaries.size())}))
    {
        TrackInitializer& init    = inits.initializers[ThreadId(
            inits.initializers.size() - primaries.size() + tid.get())];
        const Primary&    primary = primaries[tid.get()];

        // Construct a track initializer from a primary particle
        init.sim.track_id         = primary.track_id;
        init.sim.parent_id        = TrackId{};
        init.sim.event_id         = primary.event_id;
        init.sim.alive            = true;
        init.geo.pos              = primary.position;
        init.geo.dir              = primary.direction;
        init.particle.particle_id = primary.particle_id;
        init.particle.energy      = primary.energy;
    }
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
