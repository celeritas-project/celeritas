//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/distribution/MuBBEnergyDistribution.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/random/distribution/InverseSquareDistribution.hh"
#include "corecel/random/distribution/RejectionSampler.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/phys/ParticleTrackView.hh"

#include "detail/Utils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Sample delta ray energy for the muon Bethe-Bloch ionization model.
 8
 * This samples the energy according to the muon Bethe-Bloch model, as
 * described in the Geant4 Physics Reference Manual release 11.2 section 11.1.
 * At the higher energies for which this model is applied, leading radiative
 * corrections are taken into account. The differential cross section can be
 * written as
 * \f[
   \sigma(E, \epsilon) = \sigma_{BB}(E, \epsilon)\left[1 + \frac{\alpha}{2\pi}
   \log \left(1 + \frac{2\epsilon}{m_e} \log \left(\frac{4 m_e E(E -
   \epsilon}{m_{\mu}^2(2\epsilon + m_e)} \right) \right) \right].
 * \f]
 * \f$ \sigma_{BB}(E, \epsilon) \f$ is the Bethe-Bloch cross section, \f$ m_e
 * \f$ is the electron mass, \f$ m_{\mu} \f$ is the muon mass, \f$ E \f$ is the
 * total energy of the muon, and \f$ \epsilon = \omega + T \f$ is the energy
 * transfer, where \f$ T \f$ is the kinetic energy of the electron and \f$
 * \omega \f$ is the energy of the radiative gamma (which is neglected).
 *
 * As in the Bethe-Bloch model, the energy is sampled by factorizing the cross
 * section as \f$ \sigma = C f(T) g(T) \f$, where \f$ f(T) = \frac{1}{T^2} \f$
 * and \f$ T \in [T_{cut}, T_{max}] \f$. The energy is sampled from \f$ f(T)
 * \f$ and accepted with probability \f$ g(T) \f$.
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
    // Construct with incident and exiting particle data
    inline CELER_FUNCTION
    MuBBEnergyDistribution(ParticleTrackView const& particle,
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
    // Total energy of the incident particle [MeV]
    real_type total_energy_;
    // Square of fractional speed of light for incident particle
    real_type beta_sq_;
    // Secondary electron mass
    real_type electron_mass_;
    // Secondary electron cutoff energy [MeV]
    Energy min_energy_;
    // Maximum energy of the secondary electron [MeV]
    Energy max_energy_;
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
    static CELER_CONSTEXPR_FUNCTION Constant alpha_over_twopi()
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
MuBBEnergyDistribution::MuBBEnergyDistribution(ParticleTrackView const& particle,
                                               Energy electron_cutoff,
                                               Mass electron_mass)
    : inc_mass_(value_as<Mass>(particle.mass()))
    , total_energy_(value_as<Energy>(particle.total_energy()))
    , beta_sq_(particle.beta_sq())
    , electron_mass_(value_as<Mass>(electron_mass))
    , min_energy_(electron_cutoff)
    , max_energy_(detail::calc_max_secondary_energy(particle, electron_mass))
    , use_rad_correction_(particle.energy() > rad_correction_limit()
                          && max_energy_ > kin_energy_limit())
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
    InverseSquareDistribution sample_energy(value_as<Energy>(min_energy_),
                                            value_as<Energy>(max_energy_));
    real_type energy;
    real_type target;
    do
    {
        energy = sample_energy(rng);
        target = 1 - (beta_sq_ / value_as<Energy>(max_energy_)) * energy
                 + real_type(0.5) * ipow<2>(energy / total_energy_);

        if (use_rad_correction_
            && energy > value_as<Energy>(kin_energy_limit()))
        {
            real_type a1 = std::log(1 + 2 * energy / electron_mass_);
            real_type a3 = std::log(4 * total_energy_ * (total_energy_ - energy)
                                    / ipow<2>(inc_mass_));
            target *= (1 + alpha_over_twopi() * a1 * (a3 - a1));
        }
    } while (RejectionSampler<>(target, envelope_)(rng));

    CELER_ENSURE(energy > 0);
    return Energy{energy};
}

//---------------------------------------------------------------------------//
/*!
 * Calculate envelope maximum for rejection sampling of secondary energy.
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
