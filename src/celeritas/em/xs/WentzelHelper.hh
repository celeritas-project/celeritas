//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/xs/WentzelHelper.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/phys/AtomicNumber.hh"
#include "celeritas/phys/ParticleTrackView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Helper class for the Wentzel OK and VI Coulomb scattering model.
 *
 * This calculates the Moliere screening coefficient, the maximum scattering
 * angle off of electrons, and the ratio of the electron to total Wentzel cross
 * sections.
 *
 * References:
 * [PRM] Geant4 Physics Reference Manual (Release 11.1) section 8.5.
 */
class WentzelHelper
{
  public:
    //!@{
    //! \name Type aliases
    using MomentumSq = units::MevMomentumSq;
    using Energy = units::MevEnergy;
    using Mass = units::MevMass;
    //!@}

  public:
    // Construct from particle and material properties
    inline CELER_FUNCTION WentzelHelper(ParticleTrackView const& particle,
                                        AtomicNumber target_z,
                                        WentzelRef const& data,
                                        Energy cutoff);

    //! Get the target atomic number
    CELER_FUNCTION AtomicNumber atomic_number() const { return target_z_; }

    //! Get the Moliere screening coefficient
    CELER_FUNCTION real_type screening_coefficient() const
    {
        return screening_coefficient_;
    }

    //! Get the maximum scattering angle off of electrons
    CELER_FUNCTION real_type costheta_max_electron() const
    {
        return cos_t_max_elec_;
    }

    // The ratio of electron to total cross section for Coulomb scattering
    inline CELER_FUNCTION real_type calc_xs_ratio() const;

  private:
    //// DATA ////

    // Target atomic number
    AtomicNumber const target_z_;

    // Moliere screening coefficient
    real_type screening_coefficient_;

    // Cosine of the maximum scattering angle off of electrons
    real_type cos_t_max_elec_;

    //// HELPER FUNCTIONS ////

    // Calculate the Moliere screening coefficient
    inline CELER_FUNCTION real_type
    calc_screening_coefficient(ParticleTrackView const& particle) const;

    // Calculate the screening coefficient R^2 for electrons
    CELER_CONSTEXPR_FUNCTION real_type screen_r_sq_elec() const;

    // Calculate the (cosine of) the maximum scattering angle off of electrons
    inline CELER_FUNCTION real_type calc_costheta_max_electron(
        ParticleTrackView const&, WentzelRef const&, Energy) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from particle and material properties.
 */
CELER_FUNCTION
WentzelHelper::WentzelHelper(ParticleTrackView const& particle,
                             AtomicNumber target_z,
                             WentzelRef const& data,
                             Energy cutoff)
    : target_z_(target_z)
    , screening_coefficient_(this->calc_screening_coefficient(particle)
                             * data.screening_factor)
    , cos_t_max_elec_(this->calc_costheta_max_electron(particle, data, cutoff))
{
    CELER_EXPECT(screening_coefficient_ > 0);
    CELER_EXPECT(cos_t_max_elec_ >= -1 && cos_t_max_elec_ <= 1);
}

//---------------------------------------------------------------------------//
/*!
 * Ratio of electron cross section to total (nuclear + electron) cross section.
 */
CELER_FUNCTION real_type WentzelHelper::calc_xs_ratio() const
{
    // Calculating only reduced cross sections by elimination mutual factors
    // in the ratio.
    real_type nuc_xsec = target_z_.get() / (1 + screening_coefficient_);
    real_type elec_xsec = (1 - cos_t_max_elec_)
                          / (1 - cos_t_max_elec_ + 2 * screening_coefficient_);

    return elec_xsec / (nuc_xsec + elec_xsec);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the Moliere screening coefficient as in [PRM] eqn 8.51.
 */
CELER_FUNCTION real_type WentzelHelper::calc_screening_coefficient(
    ParticleTrackView const& particle) const
{
    // TODO: Reference for just proton correction?
    real_type correction = 1;
    real_type sq_cbrt_z = fastpow(real_type(target_z_.get()), real_type{2} / 3);
    if (target_z_.get() > 1)
    {
        real_type tau = value_as<Energy>(particle.energy())
                        / value_as<Mass>(particle.mass());
        // TODO: Reference for this factor?
        real_type factor = std::sqrt(tau / (tau + sq_cbrt_z));

        correction = min(target_z_.get() * real_type{1.13},
                         real_type{1.13}
                             + real_type{3.76}
                                   * ipow<2>(target_z_.get()
                                             * constants::alpha_fine_structure)
                                   * factor / particle.beta_sq());
    }

    return correction * this->screen_r_sq_elec() * sq_cbrt_z
           / value_as<MomentumSq>(particle.momentum_sq());
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the constant factor of the screening coefficient.
 *
 * This is the constant prefactor \f$ R^2 / Z^{2/3} \f$ of the screening
 * coefficient for incident electrons (equation 8.51 in [PRM]). The screening
 * radius \f$ R \f$ is given by:
 * \f[
   R = \frac{\hbar Z^{1/3}}{2C_{TF} a_0},
 * \f]
 * where the Thomas-Fermi constant \f$ C_{TF} \f$ is defined as
 * \f[
   C_{TF} = \frac{1}{2} \left(\frac{3\pi}{4}\right)^{2/3}.
 * \f]
 */
CELER_CONSTEXPR_FUNCTION real_type WentzelHelper::screen_r_sq_elec() const
{
    //! Thomas-Fermi constant \f$ C_{TF} \f$
    constexpr real_type ctf = 0.8853413770001135;

    return native_value_to<MomentumSq>(
               ipow<2>(constants::hbar_planck / (2 * ctf * constants::a0_bohr)))
        .value();
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the maximum scattering angle off the target's electrons.
 *
 * This calculates the cosine of the maximum polar angle that the incident
 * particle can scatter off of the target's electrons.
 */
CELER_FUNCTION real_type
WentzelHelper::calc_costheta_max_electron(ParticleTrackView const& particle,
                                          WentzelRef const& data,
                                          Energy cutoff) const
{
    real_type inc_energy = value_as<Energy>(particle.energy());
    real_type mass = value_as<Mass>(particle.mass());

    real_type max_energy = particle.particle_id() == data.ids.electron
                               ? real_type{0.5} * inc_energy
                               : inc_energy;
    real_type final_energy = inc_energy
                             - min(value_as<Energy>(cutoff), max_energy);

    if (final_energy > 0)
    {
        real_type incident_ratio = 1 + 2 * mass / inc_energy;
        real_type final_ratio = 1 + 2 * mass / final_energy;
        real_type cos_t_max = std::sqrt(incident_ratio / final_ratio);

        return clamp(cos_t_max, real_type{0}, real_type{1});
    }
    return 0;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
