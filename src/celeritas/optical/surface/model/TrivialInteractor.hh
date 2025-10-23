//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/TrivialInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayOperators.hh"
#include "celeritas/optical/CoreTrackView.hh"

#include "SurfaceInteraction.hh"
#include "TrivialInteractionData.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Return a trivial interaction based on the \c TrivialInteractionMode.
 *
 * Each surface in the trivial model has one interaction mode, which is applied
 * to all incident tracks. None of the modes depend on any of the previous
 * surface physics state or models.
 *
 *  1. All photons are absorbed on the surface.
 *  2. All photons are transmitted with no change to direction or polarization.
 *  3. All photons are reflected (back-scattered) with opposite direction and
 * polarization.
 */
class TrivialInteractor
{
  public:
    // Construct interactor for a specific mode
    inline CELER_FUNCTION TrivialInteractor(TrivialInteractionMode mode,
                                            Real3 const& dir,
                                            Real3 const& pol);

    // Calculate surface interaction
    inline CELER_FUNCTION SurfaceInteraction operator()() const;

  private:
    TrivialInteractionMode mode_;
    Real3 const& dir_;
    Real3 const& pol_;
};

//---------------------------------------------------------------------------//
/*!
 * Return a trivial interaction for the track based on the surface.
 */
struct TrivialInteractionExecutor
{
    NativeCRef<TrivialInteractionData> data;

    CELER_FUNCTION SurfaceInteraction operator()(CoreTrackView const& track) const
    {
        auto sub_model_id = track.surface_physics()
                                .interface(SurfacePhysicsOrder::interaction)
                                .internal_surface_id();

        CELER_ASSERT(sub_model_id < data.modes.size());

        return TrivialInteractor{data.modes[sub_model_id],
                                 track.geometry().dir(),
                                 track.particle().polarization()}();
    }
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct the interactor for the given mode, incident direction, and
 * incident polarization.
 */
CELER_FUNCTION TrivialInteractor::TrivialInteractor(TrivialInteractionMode mode,
                                                    Real3 const& dir,
                                                    Real3 const& pol)
    : mode_(mode), dir_(dir), pol_(pol)
{
}

//---------------------------------------------------------------------------//
/*!
 * Calculate surface interaction based on the interaction mode.
 */
CELER_FUNCTION SurfaceInteraction TrivialInteractor::operator()() const
{
    SurfaceInteraction result;

    switch (mode_)
    {
        case TrivialInteractionMode::absorb:
            result = SurfaceInteraction::from_absorption();
            break;
        case TrivialInteractionMode::transmit:
            result = {SurfaceInteraction::Action::refracted, dir_, pol_};
            break;
        case TrivialInteractionMode::backscatter:
            result = {SurfaceInteraction::Action::reflected,
                      negate(dir_),
                      negate(pol_)};
            break;
        default:
            CELER_ASSERT_UNREACHABLE();
    }

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
