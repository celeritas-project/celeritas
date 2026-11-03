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
 * The field propagator uses the sphere to skip boundary searches, and MSC uses
 * its radius to limit the step and to determine whether the track can be
 * displaced. This must run before either of them, while the track is at the
 * start of the step.
 *
 * The search distance is the greater of the two distances requested:
 * - a track in a field needs at most the length of the step (the physical step
 *   length is never less than the geometric path after MSC), and
 * - MSC needs the safety up to the maximum travel distance of the particle,
 *   which is usually much larger than the step.
 *
 * Tracks that need neither (a track outside the field that MSC doesn't apply
 * to) or that start on a boundary, where the safety is zero, keep their
 * previous sphere: the current position can't be its center.
 *
 * \tparam IF Function-like condition for whether a track is in a field
 */
template<class IF>
struct SafetyApplier
{
    inline CELER_FUNCTION void operator()(CoreTrackView const& track);

    NativeCRef<UrbanMscData> msc;  //!< MSC data (empty if MSC is disabled)
    IF is_in_field;
};

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//
template<class IF>
CELER_CTAD_FUNCTION SafetyApplier(NativeCRef<UrbanMscData> const&, IF&&)
    -> SafetyApplier<IF>;

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Get native MSC data for the applier, or empty data if MSC is disabled.
 */
inline NativeCRef<UrbanMscData> get_msc_ref(UrbanMscParams const* msc)
{
    return msc ? msc->ref<MemSpace::native>() : NativeCRef<UrbanMscData>{};
}

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Calculate and save the safety sphere for a track if needed.
 */
template<class IF>
CELER_FUNCTION void SafetyApplier<IF>::operator()(CoreTrackView const& track)
{
    auto sim = track.sim();
    auto geo = track.geometry();
    if (sim.status() != TrackStatus::alive || geo.is_on_boundary())
    {
        return;
    }

    real_type max_safety = 0;
    if (msc)
    {
        max_safety = UrbanMsc{msc}.max_safety(track);
    }
    if (is_in_field(track))
    {
        max_safety = celeritas::max(max_safety, sim.step_length());
    }

    if (max_safety > 0)
    {
        track.physics_step().safety_sphere(
            {geo.pos(), geo.find_safety(max_safety)});
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
