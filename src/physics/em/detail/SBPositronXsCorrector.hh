//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SBPositronXsCorrector.hh
//---------------------------------------------------------------------------//
#pragma once

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
 * exiting kinetic energy of the positron.
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
} // namespace detail
} // namespace celeritas

#include "SBPositronXsCorrector.i.hh"
