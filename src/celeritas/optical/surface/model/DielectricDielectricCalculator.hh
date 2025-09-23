//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/DielectricDielectricCalculator.hh
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
 * Calculate refracted wave between two dielectric media.
 */
class DielectricDielectricCalculator
{
  public:
    // Construct from photon and surface data
    inline CELER_FUNCTION
    DielectricDielectricCalculator(PhotonPhasor const& inc_photon,
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
CELER_FUNCTION DielectricDielectricCalculator::DielectricDielectricCalculator(
    PhotonPhasor const&, Real3 const&, real_type)
{
}

//---------------------------------------------------------------------------//
/*!
 * Calculate interaction for refracted wave.
 */
CELER_FUNCTION SurfaceInteraction DielectricDielectricCalculator::operator()() const
{
    return {};
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
