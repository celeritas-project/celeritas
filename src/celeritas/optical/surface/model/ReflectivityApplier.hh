//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/ReflectivityApplier.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/optical/CoreTrackView.hh"
#include "celeritas/optical/detail/OpticalKillTally.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Wrap a reflectivity executor and apply it to a track.
 *
 * The functor \c F must take a \c CoreTrackView and return a \c
 * ReflectivityAction.
 */
template<class F>
struct ReflectivityApplier
{
    F sample_reflectivity;

    inline CELER_FUNCTION void operator()(CoreTrackView const&) const;
};

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//
template<class F>
CELER_FUNCTION ReflectivityApplier(F&&) -> ReflectivityApplier<F>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Apply sampled reflectivity result to the track.
 */
template<class F>
CELER_FUNCTION void ReflectivityApplier<F>::operator()(
    CoreTrackView const& track) const
{
    // Sample reflectivity and set it
    auto action = this->sample_reflectivity(track);

    auto s_phys = track.surface_physics();
    s_phys.reflectivity_action(action);

#if !CELER_DEVICE_COMPILE
    if (celeritas::optical::detail::surface_trace_enabled()
        && static_cast<int>(track.track_slot_id().get())
               == celeritas::optical::detail::traced_slot().load())
    {
        char buf[160];
        std::snprintf(buf,
                      sizeof(buf),
                      "slot%d REFL action=%d pos=%u dir_n=%.3f",
                      static_cast<int>(track.track_slot_id().get()),
                      static_cast<int>(action),
                      s_phys.traversal().pos().unchecked_get(),
                      dot_product(track.geometry().dir(),
                                  s_phys.global_normal()));
        celeritas::optical::detail::trace_surface(buf);
    }
#endif

#if !CELER_DEVICE_COMPILE
    {
        // Tally every reflectivity outcome (not just the kill) so the
        // per-surface absorb fraction can be compared between geometry
        // drivers, plus the physical surface actually selected
        char const* label = action == ReflectivityAction::absorb
                                ? "refl-decide-absorb"
                                : (action == ReflectivityAction::transmit
                                       ? "refl-decide-transmit"
                                       : "refl-decide-reflect");
        bool uv = track.particle().energy().value() > 4.576e-6;
        celeritas::optical::detail::tally_optical_kill(
            label, track.geometry().volume_id().unchecked_get(), uv);
        char sbuf[64];
        std::snprintf(sbuf,
                      sizeof(sbuf),
                      "refl-physsurf-%u",
                      s_phys.interface(SurfacePhysicsOrder::reflectivity)
                          .internal_surface_id()
                          .unchecked_get());
        celeritas::optical::detail::tally_optical_kill(
            sbuf, track.geometry().volume_id().unchecked_get(), uv);
    }
#endif

    if (action == ReflectivityAction::absorb)
    {
        // Mark particle as killed
#if !CELER_DEVICE_COMPILE
        celeritas::optical::detail::tally_optical_kill(
            "reflectivity-absorbed",
            track.geometry().volume_id().unchecked_get(),
            track.particle().energy().value() > 4.576e-6);
#endif
        track.sim().status(TrackStatus::killed);
    }
    else if (action == ReflectivityAction::transmit)
    {
        // Move across the boundary
        auto traverse = s_phys.traversal();
        traverse.cross_interface(traverse.dir());
        if (traverse.is_exiting())
        {
            // End boundary crossing if exiting
            track.sim().post_step_action(s_phys.scalars().post_boundary_action);
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
