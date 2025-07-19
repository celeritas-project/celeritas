//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/distribution/BetheBlochEnergyDistribution.hh
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
 * This samples the energy according to the Bethe-Bloch model, as described in
 * the Geant4 Physics Reference Manual release 11.2 section 12.1.5. The
 * Bethe-Bloch differential cross section can be written as
 * \f[
   \difd{\sigma}{T} = 2\pi r_e^2 mc^2 Z \frac{z_p^2}{\beta^2}\frac{1}{T^2}
   \left[1 - \beta^2 \frac{T}{T_{max}} + s \frac{T^2}{2E^2} \right]
 * \f]
 * and factorized as
 * \f[
   \difd{\sigma}{T} = C f(T) g(T)
 * \f]
 * with \f$ T \in [T_{cut}, T_{max}] \f$, where \f$ f(T) = \frac{1}{T^2} \f$,
 * \f$ g(T) = 1 - \beta^2 \frac{T}{T_max} + s \frac{T^2}{2 E^2} \f$, \f$ T \f$
 * is the kinetic energy of the electron, \f$ E \f$ is the total energy of the
 * incident particle, and \f$ s \f$ is 0 for spinless particles and 1
 * otherwise. The energy is sampled from \f$ f(T) \f$ and accepted with
 * probability \f$ g(T) \f$.
 */
class BetheBlochEnergyDistribution
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
    BetheBlochEnergyDistribution(ParticleTrackView const& particle,
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
    // Incident particle mass
    real_type inc_mass_;
    // Square of fractional speed of light for incident particle
    real_type beta_sq_;
    // Minimum energy of the secondary electron [MeV]
    Energy min_energy_;
    // Maximum energy of the secondary electron [MeV]
    Energy max_energy_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with incident and exiting particle data.
 */
CELER_FUNCTION
BetheBlochEnergyDistribution::BetheBlochEnergyDistribution(
    ParticleTrackView const& particle,
    Energy electron_cutoff,
    Mass electron_mass)
    : inc_mass_(value_as<Mass>(particle.mass()))
    , beta_sq_(particle.beta_sq())
    , min_energy_(electron_cutoff)
    , max_energy_(detail::calc_max_secondary_energy(particle, electron_mass))
{
}

//---------------------------------------------------------------------------//
/*!
 * Sample secondary electron energy.
 */
template<class Engine>
CELER_FUNCTION auto BetheBlochEnergyDistribution::operator()(Engine& rng)
    -> Energy
{
    InverseSquareDistribution sample_energy(value_as<Energy>(min_energy_),
                                            value_as<Energy>(max_energy_));
    real_type energy;
    do
    {
        // Sample 1/E^2 from Emin to Emax
        energy = sample_energy(rng);
        /*!
         * \todo Adjust rejection functions if particle has positive spin
         */
    } while (RejectionSampler<>(
        1 - (beta_sq_ / value_as<Energy>(max_energy_)) * energy)(rng));

    /*!
     * \todo For hadrons, suppress high energy delta ray production with the
     * projectile form factor
     */

    return Energy{energy};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
