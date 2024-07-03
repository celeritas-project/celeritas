//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/distribution/MuBBEnergyDistribution.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/random/distribution/BernoulliDistribution.hh"
#include "celeritas/random/distribution/UniformRealDistribution.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Sample the energy of the delta ray for the MuBetheBloch ionization model.
 */
class MuBBEnergyDistribution
{
  public:
    //!@{
    //! \name Type aliases
    using Energy = units::MevEnergy;
    using Mass = units::MevMass;
    //!@}

  public:
    // Construct with data from MollerBhabhaInteractor
    inline CELER_FUNCTION MuBBEnergyDistribution(Energy inc_energy,
                                                 Mass inc_mass,
                                                 real_type beta_sq,
                                                 Mass electron_mass,
                                                 Energy electron_cutoff,
                                                 Energy max_secondary_energy);

    // Sample the exiting energy
    template<class Engine>
    inline CELER_FUNCTION Energy operator()(Engine& rng);

  private:
    //// DATA ////

    // Electron incident energy [MeV]
    real_type inc_energy_;
    // Incident partcle mass
    real_type inc_mass_;
    // Total energy of the incident particle [MeV]
    real_type total_energy_;
    // Square of fractional speed of light for incident particle
    real_type beta_sq_;
    // Secondary electron mass
    real_type electron_mass_;
    // Secondary electron cutoff energy [MeV]
    real_type electron_cutoff_;
    // Maximum energy of the secondary electron [MeV]
    real_type max_secondary_energy_;
    // Whether to apply the radiative correction
    bool use_rad_correction_;
    // Envelope distribution for rejection sampling
    real_type envelope_;

    //// CONSTANTS ////

    //! Incident energy above which the radiative correction is applied [MeV]
    static CELER_CONSTEXPR_FUNCTION Energy rad_correction_limit()
    {
        return units::MevEnergy{250};  //!< 250 MeV
    }

    //! Lower limit of radiative correction integral [MeV]
    static CELER_CONSTEXPR_FUNCTION Energy kin_energy_limit()
    {
        // This is the lower limit of the energy transfer from the incident
        // muon to the delta ray and radiative gammas
        return units::MevEnergy{0.1};  //!< 100 keV
    }

    //! Fine structure constant over two pi
    static CELER_CONSTEXPR_FUNCTION real_type alpha_over_twopi()
    {
        return constants::alpha_fine_structure / (2 * constants::pi);
    }

    //// HELPER FUNCTIONS ////

    inline CELER_FUNCTION real_type calc_envelope_distribution() const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with incident and exiting particle data.
 */
CELER_FUNCTION
MuBBEnergyDistribution::MuBBEnergyDistribution(Energy inc_energy,
                                               Mass inc_mass,
                                               real_type beta_sq,
                                               Mass electron_mass,
                                               Energy electron_cutoff,
                                               Energy max_secondary_energy)
    : inc_energy_(value_as<Energy>(inc_energy))
    , inc_mass_(value_as<Mass>(inc_mass))
    , total_energy_(inc_energy_ + inc_mass_)
    , beta_sq_(beta_sq)
    , electron_mass_(value_as<Mass>(electron_mass))
    , electron_cutoff_(value_as<Energy>(electron_cutoff))
    , max_secondary_energy_(value_as<Energy>(max_secondary_energy))
    , use_rad_correction_(
          inc_energy_ > value_as<Energy>(rad_correction_limit())
          && max_secondary_energy_ > value_as<Energy>(kin_energy_limit()))
    , envelope_(this->calc_envelope_distribution())
{
}

//---------------------------------------------------------------------------//
/*!
 * Sample secondary electron energy.
 */
template<class Engine>
CELER_FUNCTION auto MuBBEnergyDistribution::operator()(Engine& rng) -> Energy
{
    real_type energy;
    real_type target;
    do
    {
        energy = electron_cutoff_ * max_secondary_energy_
                 / UniformRealDistribution(electron_cutoff_,
                                           max_secondary_energy_)(rng);
        target = 1 - beta_sq_ * energy / max_secondary_energy_
                 + ipow<2>(energy) / (2 * ipow<2>(total_energy_));

        if (use_rad_correction_
            && energy > value_as<Energy>(kin_energy_limit()))
        {
            real_type a1 = std::log(1 + 2 * energy / electron_mass_);
            real_type a3 = std::log(4 * total_energy_ * (total_energy_ - energy)
                                    / ipow<2>(inc_mass_));
            target *= (1 + alpha_over_twopi() * a1 * (a3 - a1));
        }
    } while (!BernoulliDistribution(target / envelope_)(rng));

    return Energy{energy};
}

//---------------------------------------------------------------------------//
/*!
 * Calculate envelope distribution for rejection sampling of secondary energy.
 */
CELER_FUNCTION real_type MuBBEnergyDistribution::calc_envelope_distribution() const
{
    if (use_rad_correction_)
    {
        return 1
               + alpha_over_twopi()
                     * ipow<2>(std::log(2 * total_energy_ / inc_mass_));
    }
    return 1;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
