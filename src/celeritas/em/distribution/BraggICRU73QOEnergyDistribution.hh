//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/distribution/BraggICRU73QOEnergyDistribution.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/random/distribution/InverseSquareDistribution.hh"
#include "corecel/random/distribution/RejectionSampler.hh"
#include "celeritas/Constants.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/phys/ParticleTrackView.hh"

#include "detail/Utils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Sample the energy of the delta ray for muon or hadron ionization.
 *
 * This samples the energy according to the Bragg and ICRU73QO models, as
 * described in the Geant4 Physics Reference Manual release 11.2 section 11.1.
 */
class BraggICRU73QOEnergyDistribution
{
  public:
    //!@{
    //! \name Type aliases
    using Energy = units::MevEnergy;
    using Mass = units::MevMass;
    //!@}

  public:
    // Construct with incident and exiting particle data
    inline CELER_FUNCTION
    BraggICRU73QOEnergyDistribution(ParticleTrackView const& particle,
                                    Energy electron_cutoff,
                                    Mass electron_mass);

    // Sample the exiting energy
    template<class Engine>
    inline CELER_FUNCTION Energy operator()(Engine& rng);

    //! Minimum energy of the secondary electron [MeV].
    CELER_FUNCTION Energy min_secondary_energy() const { return min_energy_; }

    //! Maximum energy of the secondary electron [MeV].
    CELER_FUNCTION Energy max_secondary_energy() const { return max_energy_; }

  private:
    //// DATA ////

    // Incident particle mass
    real_type inc_mass_;
    // Square of fractional speed of light for incident particle
    real_type beta_sq_;
    // Minimum energy of the secondary electron [MeV]
    Energy min_energy_;
    // Maximum energy of the secondary electron [MeV]
    Energy max_energy_;

    //// CONSTANTS ////

    //! Used in Bragg model to calculate minimum energy transfer to electron
    static CELER_CONSTEXPR_FUNCTION Energy bragg_lowest_kin_energy()
    {
        return units::MevEnergy{2.5e-4};  //! 0.25 keV
    }

    //! Used in ICRU73QO model to calculate minimum energy transfer to electron
    static CELER_CONSTEXPR_FUNCTION Energy icru73qo_lowest_kin_energy()
    {
        return units::MevEnergy{5e-3};  //! 5 keV
    }
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with incident and exiting particle data.
 *
 * \todo Use proton mass from imported data instead of a constant
 */
CELER_FUNCTION
BraggICRU73QOEnergyDistribution::BraggICRU73QOEnergyDistribution(
    ParticleTrackView const& particle,
    Energy electron_cutoff,
    Mass electron_mass)
    : inc_mass_(value_as<Mass>(particle.mass()))
    , beta_sq_(particle.beta_sq())
    , min_energy_(
          min(electron_cutoff,
              Energy(value_as<Energy>(particle.charge() < zero_quantity()
                                          ? icru73qo_lowest_kin_energy()
                                          : bragg_lowest_kin_energy())
                     * inc_mass_
                     / native_value_to<Mass>(constants::proton_mass).value())))
    , max_energy_(detail::calc_max_secondary_energy(particle, electron_mass))
{
}

//---------------------------------------------------------------------------//
/*!
 * Sample secondary electron energy.
 */
template<class Engine>
CELER_FUNCTION auto BraggICRU73QOEnergyDistribution::operator()(Engine& rng)
    -> Energy
{
    InverseSquareDistribution sample_energy(value_as<Energy>(min_energy_),
                                            value_as<Energy>(max_energy_));
    real_type energy;
    do
    {
        // Sample 1/E^2 from Emin to Emax
        energy = sample_energy(rng);
    } while (RejectionSampler<>(
        1 - (beta_sq_ / value_as<Energy>(max_energy_)) * energy)(rng));

    CELER_ENSURE(energy > 0);
    return Energy{energy};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
