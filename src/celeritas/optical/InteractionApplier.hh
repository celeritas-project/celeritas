//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/InteractionApplier.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/sys/KernelTraits.hh"
#include "celeritas/geo/GeoFwd.hh"

#include "CoreTrackView.hh"
#include "Interaction.hh"
#include "ParticleTrackView.hh"
#include "SimTrackView.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Wrap an interaction executor and apply it to a track.
 *
 * The function F must take a \c CoreTrackView and return a \c Interaction
 */
template<class F>
struct InteractionApplierBaseImpl
{
    F sample_interaction;

    CELER_FUNCTION void operator()(CoreTrackView const&);
};

//---------------------------------------------------------------------------//
/*!
 * This class is partially specialized with a second template argument to
 * extract any launch bounds from the functor class.
 *
 * \todo Generalize this with the core interaction applier
 */
template<class F, typename = void>
struct InteractionApplier : public InteractionApplierBaseImpl<F>
{
    CELER_FUNCTION InteractionApplier(F&& f)
        : InteractionApplierBaseImpl<F>{celeritas::forward<F>(f)}
    {
    }
};

template<class F>
struct InteractionApplier<F, std::enable_if_t<kernel_max_blocks_min_warps<F>>>
    : public InteractionApplierBaseImpl<F>
{
    static constexpr int max_block_size = F::max_block_size;
    static constexpr int min_warps_per_eu = F::min_warps_per_eu;

    CELER_FUNCTION InteractionApplier(F&& f)
        : InteractionApplierBaseImpl<F>{celeritas::forward<F>(f)}
    {
    }
};

template<class F>
struct InteractionApplier<F, std::enable_if_t<kernel_max_blocks<F>>>
    : public InteractionApplierBaseImpl<F>
{
    static constexpr int max_block_size = F::max_block_size;

    CELER_FUNCTION InteractionApplier(F&& f)
        : InteractionApplierBaseImpl<F>{celeritas::forward<F>(f)}
    {
    }
};

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//
template<class F>
CELER_FUNCTION InteractionApplier(F&&) -> InteractionApplier<F>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Sample an interaction and apply to the track view.
 *
 * The given track *must* be an active track with the correct step limit action
 * ID.
 */
template<class F>
CELER_FUNCTION void
InteractionApplierBaseImpl<F>::operator()(CoreTrackView const& track)
{
    Interaction result = this->sample_interaction(track);

    // Currently we have no optical actions capable of failing. This can be
    // added in the future as necessary.
    CELER_ASSERT(result.action != Interaction::Action::failed);

    if (result.action == Interaction::Action::absorbed)
    {
        // Mark particle as killed
        track.sim().status(TrackStatus::killed);
    }
    else
    {
        // Update direction and polarization
        track.geometry().set_dir(result.direction);
        track.particle().polarization(result.polarization);
    }
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
