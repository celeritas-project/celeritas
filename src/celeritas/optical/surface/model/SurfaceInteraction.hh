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
 * Representation of a direction and polarization as a phasor (phase vector).
 */
struct PhotonPhasor
{
    Real3 direction;
    Real3 polarization;

    //! Whether the phasor is a valid photon state
    CELER_FUNCTION bool is_valid() const
    {
        return is_soft_unit_vector(direction)
               && is_soft_unit_vector(polarization)
               && soft_zero(dot_product(direction, polarization));
    }
};

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
    PhotonPhasor photon;  //!< Post-interaction photon state

    //! Return an interaction representing an absorbed photon
    static inline CELER_FUNCTION SurfaceInteraction from_absorption();

    //! Whether data is assigned and valid
    CELER_FUNCTION bool is_valid() const
    {
        return action == Action::absorbed || photon.is_valid();
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
