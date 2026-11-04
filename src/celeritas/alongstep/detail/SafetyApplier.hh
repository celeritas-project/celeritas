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
 * The search distance is the greater of the two distances requested:
 * - MSC needs the safety up to the maximum travel distance of the particle,
 *   which is usually much larger than the step, and
 * - a track that uses the sphere in a field needs at most the length of the
 *   step (the physical step length is never less than the geometric path after
 *   MSC).
 *
 * Tracks that need neither, or that start on a boundary, keep their previous
 * sphere.
 *
 * \tparam UF Function-like condition for whether a track's field propagation
 * uses the sphere
 */
template<class UF>
struct SafetyApplier
{
    inline CELER_FUNCTION void operator()(CoreTrackView const& track);

    NativeCRef<UrbanMscData> msc;  //!< MSC data (empty if MSC is disabled)
    UF uses_field_safety;
};

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//
template<class UF>
CELER_CTAD_FUNCTION SafetyApplier(NativeCRef<UrbanMscData> const&, UF&&)
    -> SafetyApplier<UF>;

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
template<class UF>
CELER_FUNCTION void SafetyApplier<UF>::operator()(CoreTrackView const& track)
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
    if (uses_field_safety(track))
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
