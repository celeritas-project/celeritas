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
struct InteractionApplier
{
    F sample_interaction;

    CELER_FUNCTION void operator()(CoreTrackView const&);
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
InteractionApplier<F>::operator()(CoreTrackView const& track)
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
    else if (result.action == Interaction::Action::scattered)
    {
        // Update direction and polarization
        track.geometry().set_dir(result.direction);
        track.particle().polarization(result.polarization);
    }
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
