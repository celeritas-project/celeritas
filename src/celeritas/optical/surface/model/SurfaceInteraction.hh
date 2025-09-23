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
 * Brief class description.
 *
 * Optional detailed class description, and possibly example usage:
 * \code
    SurfaceInteraction ...;
   \endcode
 */

struct PhotonPhasor
{
    Real3 direction;
    Real3 polarization;

    //! Whether the phaser is a valid photon state
    explicit CELER_FUNCTION operator bool() const
    {
        return is_soft_unit_vector(direction)
               && is_soft_unit_vector(polarization)
               && soft_zero(dot_product(direction, polarization));
    }
};

struct SurfaceInteraction
{
    enum Action
    {
        absorb,
        reflect,
        refract
    };

    Action action{absorb};
    PhotonPhasor photon;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
