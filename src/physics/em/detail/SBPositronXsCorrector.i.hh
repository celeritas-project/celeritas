//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SBPositronXsCorrector.i.hh
//---------------------------------------------------------------------------//
#include <cmath>

#include "base/Constants.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with positron data and energy range.
 */
SBPositronXsCorrector::SBPositronXsCorrector(Mass               positron_mass,
                                             const ElementView& el,
                                             Energy min_gamma_energy,
                                             Energy inc_energy)
    : positron_mass_{positron_mass.value()}
    , alpha_z_{celeritas::constants::alpha_fine_structure * el.atomic_number()}
    , inc_energy_(inc_energy.value())
    , cutoff_invbeta_{this->calc_invbeta(min_gamma_energy.value())}
{
    CELER_EXPECT(inc_energy > min_gamma_energy);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate scaling factor for the given exiting gamma energy.
 */
CELER_FUNCTION real_type SBPositronXsCorrector::operator()(Energy energy) const
{
    CELER_EXPECT(energy > zero_quantity());
    CELER_EXPECT(energy.value() < inc_energy_);
    real_type delta = cutoff_invbeta_ - this->calc_invbeta(energy.value());
    return std::exp(alpha_z_ * delta);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the inverse of the relativistic positron speed.
 *
 * The input here is the exiting gamma energy, so the positron energy is the
 * remainder from the incident energy. The relativistic speed \f$ \beta \f$
 * is:
 * \f[
  \beta^{-1}(K)
   = \frac{K + m c^2}{\sqrt{K (K + 2 m c^2)}}
   = \frac{K + mc^2}{\sqrt{K^2 + 2 K mc^2 + (mc^2)^2 - (mc^2)^2}}
   = \frac{K + mc^2}{\sqrt{(K + mc^2)^2 - mc^2}}
   = 1/\sqrt{1 - \left( \frac{mc^2}{K + mc^2} \right)^2}
   = 1 / \beta(K)
 * \f]
 *
 * \todo I originally wanted all these sort of calculations to be in
 * ParticleTrackView, but that class requires a full set of state, params, etc.
 * Maybe we need to refactor it so that this calculation doesn't get duplicated
 * everywhere inside the physics -- maybe a "LocalParticle" that has the same
 * functions.
 */
CELER_FUNCTION real_type
SBPositronXsCorrector::calc_invbeta(real_type gamma_energy) const
{
    CELER_EXPECT(gamma_energy > 0 && gamma_energy < inc_energy_);
    // Positron has all the energy except what it gave to the gamma
    real_type energy = inc_energy_ - gamma_energy;
    return (energy + positron_mass_)
           / std::sqrt(energy * (energy + 2 * positron_mass_));
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
