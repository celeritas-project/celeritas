//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SBPositronXsCorrector.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "base/Constants.hh"
#include "base/Types.hh"
#include "physics/base/Units.hh"
#include "physics/material/ElementView.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Scale SB differential cross sections for positrons.
 *
 * This cross section correction factor appears in the bowels of \c
 * G4SeltzerBergerModel::SampleEnergyTransfer in Geant4 and scales the SB cross
 * section by a factor
 * \f[
   \frac{\sigma_\mathrm{corrected}}{\sigma}
       = \exp( \alpha Z [ \beta^{-1}(E - E') - \beta^{-1}(E - k_c) ] )
 * \f]
 * where the inverse positron speed is : \f[
  \beta^{-1}(E) = \frac{c}{v} = \sqrt{1 - \left( \frac{m_e c^2}{E + m_e c^2}
  \right)^2}
  \,,
 * \f]
 * \f$ \alpha \f$ is the fine structure constant, \f$E\f$ is the
 * incident positron kinetic energy, \f$k_c\f$ is the gamma
 * production cutoff energy, and \f$ E' \f$ is the provisionally sampled
 * exiting kinetic energy of the photon.
 *
 * The correction factor is described in:
 *
 *   Kim, Longhuan, R. H. Pratt, S. M. Seltzer, and M. J. Berger. “Ratio of
 *   Positron to Electron Bremsstrahlung Energy Loss: An Approximate Scaling
 *   Law.” Physical Review A 33, no. 5 (May 1, 1986): 3002–9.
 *   https://doi.org/10.1103/PhysRevA.33.3002.
 *
 * \todo Integrate into the actual sampling process.
 */
class SBPositronXsCorrector
{
  public:
    //!@{
    using Energy = units::MevEnergy;
    using Mass   = units::MevMass;
    //!@}

  public:
    // Construct with positron data
    inline CELER_FUNCTION SBPositronXsCorrector(Mass positron_mass,
                                                const ElementView& el,
                                                Energy min_gamma_energy,
                                                Energy inc_energy);

    // Calculate cross section scaling factor for the given exiting energy
    inline CELER_FUNCTION real_type operator()(Energy energy) const;

  private:
    //// DATA ////

    const real_type positron_mass_;
    const real_type alpha_z_;
    const real_type inc_energy_;
    const real_type cutoff_invbeta_;

    //// HELPER FUNCTIONS ////

    inline CELER_FUNCTION real_type calc_invbeta(real_type gamma_energy) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with positron data and energy range.
 */
CELER_FUNCTION
SBPositronXsCorrector::SBPositronXsCorrector(Mass               positron_mass,
                                             const ElementView& el,
                                             Energy min_gamma_energy,
                                             Energy inc_energy)
    : positron_mass_{positron_mass.value()}
    , alpha_z_{2 * constants::pi * celeritas::constants::alpha_fine_structure
               * el.atomic_number()}
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
