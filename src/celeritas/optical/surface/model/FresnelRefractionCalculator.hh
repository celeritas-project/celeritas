//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/FresnelRefractionCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "celeritas/optical/Types.hh"

#include "SurfaceInteraction.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Calculate refracted wave between two dielectric media analytically from
 * Fresnel equations.
 */
class FresnelRefractionCalculator
{
  public:
    // Construct from photon and surface data
    inline CELER_FUNCTION
    FresnelRefractionCalculator(PhotonPhasor const& inc_photon,
                                Real3 const& normal,
                                real_type relative_r_index);

    // Calculate interaction for refracted wave
    inline CELER_FUNCTION SurfaceInteraction operator()() const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from photon and surface data.
 */
CELER_FUNCTION
FresnelRefractionCalculator::FresnelRefractionCalculator(PhotonPhasor const&,
                                                         Real3 const&,
                                                         real_type)
{
}

//---------------------------------------------------------------------------//
/*!
 * Calculate interaction for refracted wave.
 */
CELER_FUNCTION SurfaceInteraction FresnelRefractionCalculator::operator()() const
{
    return {};
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
