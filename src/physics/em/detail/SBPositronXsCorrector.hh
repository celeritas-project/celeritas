//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
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
 * This correction factor appears in the bowels of \c
 * G4SeltzerBergerModel::SampleEnergyTransfer in Geant4 and scales the SB cross
 * section by a factor
 * \f[
   \frac{\sigma_\mathrm{corrected}}{\sigma}
       = \alpha Z [ \beta(E) - \beta(E_c) ]
 \f]
 * where \f$ \beta \f$ is the Lorentz factor: \f[
  \beta(E) = \frac{E + m_e c^2}{\sqrt{E * E + 2 * E * m_e * c^2}} \,,
 * \f]
 * \f$ \alpha \f$ is the fine structure constant, \f$E_c\f$ is the gamma
 * production cutoff energy, and \f$ E \f$ is the provisionally sampled exiting
 * energy.
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
    const real_type positron_mass_;
    const real_type alpha_z_;
    const real_type inc_energy_;
    const real_type cutoff_lorentz_;

    inline CELER_FUNCTION real_type
    calc_lorentz_factor(real_type gamma_energy) const;
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

#include "SBPositronXsCorrector.i.hh"
