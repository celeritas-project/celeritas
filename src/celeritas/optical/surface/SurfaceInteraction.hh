//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/SurfaceInteraction.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "geocel/Types.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Store results from a boundary crossing interaction step.
 *
 * The \c Action enum determines whether the photon has been absorbed on the
 * surface, if it has crossed the boundary, or has remained in the initial
 * boundary.
 */
struct SurfaceInteraction
{
    enum class Action
    {
        absorbed,  //!< Absorbed on the surface
        transmitted,  //!< Crossed the boundary
        reflected,  //!< Has not crossed the boudnary
    };

    Real3 direction;  //!< Post-interaction direction
    Real3 polarization;  //!< Post-interaction polarization
    Action action{Action::reflected};  //!< Flags for interaction result

    //! Return an interaction respresenting an absorbed process
    static inline CELER_FUNCTION SurfaceInteraction from_absorption();
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct an interaction for an optical photon absorbed on the surface.
 */
CELER_FUNCTION SurfaceInteraction SurfaceInteraction::from_absorption()
{
    SurfaceInteraction result;
    result.action = Action::absorbed;
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
