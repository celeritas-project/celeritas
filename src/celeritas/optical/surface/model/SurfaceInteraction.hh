//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/SurfaceInteraction.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/math/ArrayUtils.hh"
#include "corecel/math/SoftEqual.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Result of a surface physics interaction.
 */
struct SurfaceInteraction
{
    //! Interaction result category
    enum class Action
    {
        absorbed,
        reflected,
        refracted
    };

    Action action{Action::absorbed};  //!< Flags for interaction result
    Real3 direction;
    Real3 polarization;

    //! Return an interaction representing an absorbed photon
    static inline CELER_FUNCTION SurfaceInteraction from_absorption();

    //! Whether data is assigned and valid
    CELER_FUNCTION bool is_valid() const
    {
        return action == Action::absorbed
               || (is_soft_unit_vector(direction)
                   && is_soft_unit_vector(polarization)
                   && is_soft_orthogonal(direction, polarization));
    }
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct a surface interaction for an optical photon absorbed on the
 * surface.
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
