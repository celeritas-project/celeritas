//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/FresnelRefractionCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "FresnelUtils.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Calculate refracted wave between two dielectric media analytically from
 * Fresnel equations.
 */
struct FresnelRefractionCalculator
{
    PhotonPhasor const& inc_photon;
    Real3 const& normal;
    real_type relative_r_index;

    // Calculate interaction for refracted wave
    inline CELER_FUNCTION SurfaceInteraction operator()() const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Calculate interaction for refracted wave.
 */
CELER_FUNCTION SurfaceInteraction FresnelRefractionCalculator::operator()() const
{
    FresnelCalculator calc{inc_photon, normal, relative_r_index};
    CELER_ASSERT(!calc.is_total_internal_reflection());

    SurfaceInteraction result;
    result.action = SurfaceInteraction::Action::refracted;
    result.photon.direction = calc.refracted_direction();

    result.photon.polarization = {0, 0, 0};
    axpy(calc.calc_transmission_te() * calc.inc_te_component(),
         calc.te_axis(),
         &result.photon.polarization);
    axpy(calc.calc_transmission_tm() * calc.inc_tm_component(),
         calc.tm_axis(result.photon.direction),
         &result.photon.polarization);
    result.photon.polarization = make_unit_vector(result.photon.polarization);

    CELER_ENSURE(result.is_valid());

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
