//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LivermorePEMicroXsCalculator.i.hh
//---------------------------------------------------------------------------//

#include "base/Algorithms.hh"
#include "physics/grid/GenericXsCalculator.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
CELER_FUNCTION LivermorePEMicroXsCalculator::LivermorePEMicroXsCalculator(
    const LivermorePEPointers& shared, Energy energy)
    : shared_(shared), inc_energy_(energy.value())
{
}

//---------------------------------------------------------------------------//
/*!
 * Compute cross section
 */
CELER_FUNCTION
real_type LivermorePEMicroXsCalculator::operator()(ElementId el_id) const
{
    CELER_EXPECT(el_id);
    const LivermoreElement& el     = shared_.xs_data.elements[el_id];
    const auto&             shells = shared_.xs_data.shells[el.shells];

    // In Geant4, if the incident gamma energy is below the lowest binding
    // energy, it is set to the binding energy so that the photoelectric cross
    // section is constant rather than zero for low energy gammas.
    Energy    energy     = max(inc_energy_, shells.back().binding_energy);
    real_type inv_energy = 1. / energy.value();

    real_type result = 0.;
    if (energy >= el.thresh_lo)
    {
        // Fit parameters from the final shell are used to calculate the cross
        // section integrated over all subshells
        const auto& shell = shells.back();
        const auto& param = energy >= el.thresh_hi
                                ? shared_.xs_data.reals[shell.param_hi]
                                : shared_.xs_data.reals[shell.param_lo];

        // Use the parameterization of the integrated subshell cross sections
        // clang-format off
        result
            = inv_energy * (param[0] + inv_energy * (param[1]
            + inv_energy * (param[2] + inv_energy * (param[3]
            + inv_energy * (param[4] + inv_energy * param[5])))));
        // clang-format on
    }
    else if (energy >= shells.front().binding_energy)
    {
        // Use tabulated cross sections above K-shell energy but below energy
        // limit for parameterization
        GenericXsCalculator calc_xs(el.xs_hi, shared_.xs_data.reals);
        result = ipow<3>(inv_energy) * calc_xs(energy.value());
    }
    else
    {
        // Use tabulated cross sections below K-shell energy
        GenericXsCalculator calc_xs(el.xs_lo, shared_.xs_data.reals);
        result = ipow<3>(inv_energy) * calc_xs(energy.value());
    }
    return result;
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
