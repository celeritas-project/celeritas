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
#include "celeritas/em/data/CommonCoulombData.hh"
#include "celeritas/em/data/WentzelOKVIData.hh"
#include "celeritas/mat/MaterialView.hh"
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
    using Charge = units::ElementaryCharge;
    using Energy = units::MevEnergy;
    using Mass = units::MevMass;
    using MomentumSq = units::MevMomentumSq;
    //!@}

  public:
    // Construct from particle and material properties
    inline CELER_FUNCTION WentzelHelper(ParticleTrackView const& particle,
                                        MaterialView const& material,
                                        AtomicNumber target_z,
                                        CoulombParameters const& data,
                                        CoulombIds const& ids,
                                        Energy cutoff);

    //! Get the target atomic number
    CELER_FUNCTION AtomicNumber atomic_number() const { return target_z_; }

    //! Get the Moliere screening coefficient
    CELER_FUNCTION real_type screening_coefficient() const
    {
        return screening_coefficient_;
    }

    //! Get the Mott factor
    CELER_FUNCTION real_type mott_factor() const { return mott_factor_; }

    //! Get the multiplicative factor for the cross section
    CELER_FUNCTION real_type kin_factor() const { return kin_factor_; }

    //! Get the maximum scattering angle off of electrons
    CELER_FUNCTION real_type costheta_max_electron() const
    {
        return costheta_max_elec_;
    }

    //! Get the maximum scattering angle off of a nucleus
    CELER_FUNCTION real_type costheta_max_nuclear() const
    {
        return costheta_max_nuc_;
    }

    // The ratio of electron to total cross section for Coulomb scattering
    inline CELER_FUNCTION real_type calc_xs_ratio(real_type costheta_min,
                                                  real_type costheta_max) const;

    // Calculate the electron cross section for Coulomb scattering
    inline CELER_FUNCTION real_type
    calc_xs_electron(real_type costheta_min, real_type costheta_max) const;

    // Calculate the nuclear cross section for Coulomb scattering
    inline CELER_FUNCTION real_type
    calc_xs_nuclear(real_type costheta_min, real_type costheta_max) const;

  private:
    //// DATA ////

    AtomicNumber const target_z_;
    real_type screening_coefficient_;
    real_type kin_factor_;
    real_type mott_factor_;
    real_type costheta_max_elec_;
    real_type costheta_max_nuc_;

    //// HELPER FUNCTIONS ////

    // Calculate the Moliere screening coefficient
    inline CELER_FUNCTION real_type
    calc_screening_coefficient(ParticleTrackView const& particle) const;

    // Calculate the screening coefficient R^2 for electrons
    CELER_CONSTEXPR_FUNCTION real_type screen_r_sq_elec() const;

    // Calculate the multiplicative factor for the cross section
    inline CELER_FUNCTION real_type
    calc_kin_factor(ParticleTrackView const&) const;

    // Calculate the (cosine of) the maximum scattering angle off of electrons
    inline CELER_FUNCTION real_type calc_costheta_max_electron(
        ParticleTrackView const&, CoulombIds const&, Energy) const;

    // Calculate the (cosine of) the maximum scattering angle off of a nucleus
    inline CELER_FUNCTION real_type
    calc_costheta_max_nuclear(ParticleTrackView const&,
                              MaterialView const& material,
                              CoulombParameters const&) const;

    // Calculate the common factor in the electron and nuclear cross section
    inline CELER_FUNCTION real_type calc_xs_factor(real_type costheta_min,
                                                   real_type costheta_max) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from particle and material properties.
 */
CELER_FUNCTION
WentzelHelper::WentzelHelper(ParticleTrackView const& particle,
                             MaterialView const& material,
                             AtomicNumber target_z,
                             CoulombParameters const& data,
                             CoulombIds const& ids,
                             Energy cutoff)
    : target_z_(target_z)
    , screening_coefficient_(this->calc_screening_coefficient(particle)
                             * data.screening_factor)
    , kin_factor_(this->calc_kin_factor(particle))
    , mott_factor_(particle.particle_id() == ids.electron
                       ? 1 + real_type(2e-4) * ipow<2>(target_z_.get())
                       : 1)
    , costheta_max_elec_(
          this->calc_costheta_max_electron(particle, ids, cutoff))
    , costheta_max_nuc_(
          this->calc_costheta_max_nuclear(particle, material, data))
{
    CELER_EXPECT(screening_coefficient_ > 0);
    CELER_EXPECT(costheta_max_elec_ >= -1 && costheta_max_elec_ <= 1);
    CELER_EXPECT(costheta_max_nuc_ >= -1 && costheta_max_nuc_ <= 1);
}

//---------------------------------------------------------------------------//
/*!
 * Ratio of electron cross section to total (nuclear + electron) cross section.
 */
CELER_FUNCTION real_type WentzelHelper::calc_xs_ratio(
    real_type costheta_min, real_type costheta_max) const
{
    real_type xs_elec = this->calc_xs_electron(costheta_min, costheta_max);
    return xs_elec
           / (xs_elec + this->calc_xs_nuclear(costheta_min, costheta_max));
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the electron cross section for Coulomb scattering.
 */
CELER_FUNCTION real_type WentzelHelper::calc_xs_electron(
    real_type costheta_min, real_type costheta_max) const
{
    costheta_min = max(costheta_min, costheta_max_elec_);
    costheta_max = max(costheta_max, costheta_max_elec_);
    if (costheta_min <= costheta_max)
    {
        return 0;
    }
    return this->calc_xs_factor(costheta_min, costheta_max);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the nuclear cross section for Coulomb scattering.
 */
CELER_FUNCTION real_type WentzelHelper::calc_xs_nuclear(
    real_type costheta_min, real_type costheta_max) const
{
    return target_z_.get() * this->calc_xs_factor(costheta_min, costheta_max);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the common factor in the electron and nuclear cross section.
 */
CELER_FUNCTION real_type WentzelHelper::calc_xs_factor(
    real_type costheta_min, real_type costheta_max) const
{
    return kin_factor_ * mott_factor_ * (costheta_min - costheta_max)
           / ((1 - costheta_min + 2 * screening_coefficient_)
              * (1 - costheta_max + 2 * screening_coefficient_));
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the Moliere screening coefficient as in [PRM] eqn 8.51.
 *
 * \note The \c screenZ in Geant4 is equal to twice the screening coefficient.
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
 * Calculate the multiplicative factor for the cross section.
 *
 * This calculates the factor
 * \f[
   f = \frac{2 \pi m_e^2 r_e^2 Z q^2}{\beta^2 p^2},
 * \f]
 * where \f$ m_e, r_e, Z, q, \beta \f$, and \f$ p \f$ are the electron mass,
 * classical electron radius, atomic number of the target atom, charge,
 * relativistic speed, and momentum of the incident particle, respectively.
 */
CELER_FUNCTION real_type
WentzelHelper::calc_kin_factor(ParticleTrackView const& particle) const
{
    real_type constexpr twopi_mrsq
        = 2 * constants::pi
          * ipow<2>(native_value_to<Mass>(constants::electron_mass).value()
                    * constants::r_electron);

    return twopi_mrsq * target_z_.get()
           * ipow<2>(value_as<Charge>(particle.charge()))
           / (particle.beta_sq()
              * value_as<MomentumSq>(particle.momentum_sq()));
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the maximum scattering angle off the target's electrons.
 *
 * This calculates the cosine of the maximum polar angle that the incident
 * particle can scatter off of the target's electrons.
 */
CELER_FUNCTION real_type WentzelHelper::calc_costheta_max_electron(
    ParticleTrackView const& particle, CoulombIds const& ids, Energy cutoff) const
{
    real_type inc_energy = value_as<Energy>(particle.energy());
    real_type mass = value_as<Mass>(particle.mass());

    real_type max_energy = particle.particle_id() == ids.electron
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
/*!
 * Calculate the maximum scattering angle off the target nucleus.
 */
CELER_FUNCTION real_type
WentzelHelper::calc_costheta_max_nuclear(ParticleTrackView const& particle,
                                         MaterialView const& material,
                                         CoulombParameters const& data) const
{
    if (data.is_combined)
    {
        return max(data.costheta_limit,
                   1
                       - data.a_sq_factor * material.inv_mass_cbrt_sq()
                             / value_as<MomentumSq>(particle.momentum_sq()));
    }
    return data.costheta_limit;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
