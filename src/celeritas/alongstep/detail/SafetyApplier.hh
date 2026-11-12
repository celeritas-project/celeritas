//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/alongstep/detail/SafetyApplier.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/em/data/UrbanMscData.hh"
#include "celeritas/em/msc/UrbanMsc.hh"
#include "celeritas/em/params/UrbanMscParams.hh"
#include "celeritas/global/CoreTrackView.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Calculate the safety sphere at the start of the step.
 *
 * MSC uses the radius to limit the step and to determine whether the track can
 * be displaced, and the field propagator can use the sphere to skip boundary
 * searches. This must run before either of them, while the track is at the
 * start of the step.
 *
 * The search distance is the step length or, if MSC applies to the track, the
 * maximum travel distance of the particle if that is greater. The step length
 * is conservative for field propagation since it is the physical step, and the
 * geometric path after MSC is never longer.
 *
 * Every live track gets a sphere centered on its current position, so no
 * stale sphere from an earlier step is left behind. Tracks that start on a
 * boundary, where the safety is zero, get an empty sphere.
 */
struct SafetyApplier
{
    inline CELER_FUNCTION void operator()(CoreTrackView const& track);

    NativeCRef<UrbanMscData> msc;  //!< MSC data (empty if MSC is disabled)
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Get MSC data for the applier, or empty data if MSC is disabled.
 *
 * This is called from host code in both host and device sources, where \c
 * MemSpace::native differs. The memory space must be a template parameter:
 * otherwise the two versions, which differ only by return type, share a symbol
 * and the linker keeps just one of them.
 */
template<MemSpace M>
inline UrbanMscData<Ownership::const_reference, M> get_msc_ref(
    UrbanMscParams const* msc)
{
    using RefT = UrbanMscData<Ownership::const_reference, M>;
    return msc ? msc->ref<M>() : RefT{};
}

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Calculate and save the safety sphere for a track.
 */
CELER_FUNCTION void SafetyApplier::operator()(CoreTrackView const& track)
{
    auto sim = track.sim();
    if (sim.status() != TrackStatus::alive)
    {
        return;
    }

    auto geo = track.geometry();
    SafetySphere sphere{geo.pos(), 0};
    if (!geo.is_on_boundary())
    {
        real_type max_safety = sim.step_length();
        if (msc)
        {
            max_safety
                = celeritas::max(max_safety, UrbanMsc{msc}.max_safety(track));
        }
        if (max_safety > 0)
        {
            sphere.radius = geo.find_safety(max_safety);
        }
    }
    track.physics_step().safety_sphere(sphere);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
