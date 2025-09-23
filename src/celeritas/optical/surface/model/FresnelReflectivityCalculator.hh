//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/FresnelReflectivityCalculator.hh
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
 * Calculate reflectivity analytically from Fresnel equations.
 *
 * Separately calculates the TE and TM polarization reflectivities then
 * combines them to calculate the total reflectivity for a linearly polarized
 * photon. Currently only handles real refractive indices.
 *
 * Relative refractive index is post-volume divided by pre-volume refractive
 * indices.
 */
class FresnelReflectivityCalculator
{
  public:
    // Construct from photon and surface data
    inline CELER_FUNCTION
    FresnelReflectivityCalculator(PhotonPhasor const& inc_photon,
                                  Real3 const& normal,
                                  real_type relative_r_index);

    // Calculate reflectivity
    inline CELER_FUNCTION real_type operator()() const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from photon and surface data.
 */
CELER_FUNCTION
FresnelReflectivityCalculator::FresnelReflectivityCalculator(
    PhotonPhasor const&, Real3 const&, real_type)
{
}

//---------------------------------------------------------------------------//
/*!
 * Calculate total reflectivity from Fresnel equations.
 *
 * The reflectivity is a probability to reflect in the range [0,1].
 */
CELER_FUNCTION real_type FresnelReflectivityCalculator::operator()() const
{
    return 0;
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
