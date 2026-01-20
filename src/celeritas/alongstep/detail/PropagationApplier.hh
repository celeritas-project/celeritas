//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/alongstep/detail/PropagationApplier.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "celeritas/global/CoreTrackView.hh"

#if !CELER_DEVICE_COMPILE
#    include "corecel/io/Logger.hh"
#    include "corecel/io/Repr.hh"
#endif

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Apply propagation over the step.
 *
 * \tparam TP Track propagator
 *
 * TP should be a function-like object:
 * \code Propagation (*)(CoreTrackView const&) \endcode
 */
template<class TP>
struct PropagationApplier
{
    inline CELER_FUNCTION void operator()(CoreTrackView& track);

    TP propagate;
};

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//
template<class TP>
CELER_FUNCTION PropagationApplier(TP&&) -> PropagationApplier<TP>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
template<class TP>
CELER_FUNCTION void PropagationApplier<TP>::operator()(CoreTrackView& track)
{
    auto sim = track.sim();
    CELER_EXPECT(sim.step_length() > 0);

    Propagation p;
    {
#if CELERITAS_DEBUG
        Real3 const orig_pos = track.geometry().pos();
#endif
        p = this->propagate(track);
        CELER_ASSERT(p.distance > 0);
#if CELERITAS_DEBUG
        if (CELER_UNLIKELY(track.geometry().pos() == orig_pos))
        {
            // This unusual case happens when the step length is less than
            // machine epsilon compared to the actual position. This case seems
            // to occur in vecgeom when "stuck" on a boundary, and when in a
            // field taking a small step when the track's position has a large
            // magnitude.
#    if !CELER_DEVICE_COMPILE
            CELER_LOG_LOCAL(error)
                << "Propagation of step length " << repr(sim.step_length())
                << " due to post-step action "
                << sim.post_step_action().unchecked_get()
                << " leading to distance " << repr(p.distance)
                << (p.boundary  ? " (boundary hit)"
                    : p.looping ? " (**LOOPING**)"
                                : "")
                << " failed to change position";
#    endif
            track.apply_errored();
        }
#endif
    }

    if (p.boundary)
    {
        // Stopped at a geometry boundary: this is the new step action.
        CELER_ASSERT(p.distance <= sim.step_length());
        sim.step_length(p.distance);
        sim.post_step_action(track.boundary_action());
    }
    else if (!p.looping && p.distance < sim.step_length())
    {
        // Some tracks may get stuck on a boundary and fail to move at
        // all in the field propagator, and will get bumped a small
        // distance. This primarily occurs with reentrant tracks on a
        // boundary with VecGeom.
        sim.step_length(p.distance);
        sim.post_step_action(track.propagation_limit_action());
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
