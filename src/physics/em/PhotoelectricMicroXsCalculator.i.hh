//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PhotoelectricMicroXsCalculator.i.hh
//---------------------------------------------------------------------------//

#include "base/Algorithms.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
CELER_FUNCTION PhotoelectricMicroXsCalculator::PhotoelectricMicroXsCalculator(
    const PhotoelectricInteractorPointers& shared,
    const ParticleTrackView&               particle)
    : shared_(shared), inc_energy_(particle.energy().value())
{
    REQUIRE(particle.def_id() == shared_.gamma_id);
}

//---------------------------------------------------------------------------//
/*!
 * Compute cross section
 */
CELER_FUNCTION
real_type PhotoelectricMicroXsCalculator::operator()(ElementDefId el_id) const
{
    REQUIRE(el_id);
    const LivermoreData& el = shared_.elements[el_id.get()];

    // If the incident gamma energy is below the lowest binding energy,
    // set it to the binding energy so that the photoelectric cross section
    // is constant rather than zero for low energy gammas.
    MevEnergy energy
        = max(inc_energy_, el.shells[el.shells.size() - 1].binding_energy);
    real_type inv_energy = 1. / energy.value();

    real_type result = 0.;
    if (energy >= el.thresh_low)
    {
        // Fit parameters from the final shell are used to calculate subshell
        // cross sections integrated over all shells
        const auto& shell = el.shells[el.shells.size() - 1];
        const auto& param = energy >= el.thresh_high ? shell.param_high
                                                     : shell.param_low;

        // Use the parameterization of the integrated subshell cross sections
        // clang-format off
        result
            = inv_energy * (param[0] + inv_energy * (param[1]
            + inv_energy * (param[2] + inv_energy * (param[3]
            + inv_energy * (param[4] + inv_energy * param[5])))));
        // clang-format on
    }
    else if (energy >= el.shells[0].binding_energy)
    {
        // Use tabulated cross sections above K-shell energy
        // TODO: calculate PE total xs
        // result = ipow<3>(inv_energy) * el.xs_high(energy);
    }
    else
    {
        // Use tabulated cross sections below K-shell energy
        // TODO: calculate PE total xs
        // result = ipow<3>(inv_energy) * el.xs_low(energy);
    }
    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
