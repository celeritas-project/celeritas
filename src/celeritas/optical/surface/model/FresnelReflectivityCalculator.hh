//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/FresnelReflectivityCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "FresnelUtils.hh"

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
struct FresnelReflectivityCalculator
{
    PhotonPhasor const& inc_photon;
    Real3 const& normal;
    real_type relative_r_index;

    // Calculate reflectivity
    inline CELER_FUNCTION real_type operator()() const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Calculate total reflectivity from Fresnel equations.
 *
 * The reflectivity is a probability to reflect in the range [0,1].
 */
CELER_FUNCTION real_type FresnelReflectivityCalculator::operator()() const
{
    detail::FresnelCalculator calc{inc_photon, normal, relative_r_index};
    real_type te_comp_sq = ipow<2>(calc.inc_te_component());
    real_type tm_comp_sq = ipow<2>(calc.inc_tm_component());
    real_type total_reflectivity
        = (te_comp_sq * ipow<2>(calc.calc_reflectivity_te())
           + tm_comp_sq * ipow<2>(calc.calc_reflectivity_tm()))
          / (te_comp_sq + tm_comp_sq);

    CELER_ENSURE(0 <= total_reflectivity && total_reflectivity <= 1);

    return total_reflectivity;
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
