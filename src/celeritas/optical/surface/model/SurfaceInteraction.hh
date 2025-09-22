//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/SurfaceInteraction.hh
//---------------------------------------------------------------------------//
#pragma once

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

struct PhotonState
{
    Real3 direction;
    Real3 polarization;
};

struct SurfaceInteraction
{
    bool crossed_surface{false};
    PhotonState state;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
