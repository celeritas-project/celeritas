//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
 * The Moliere screening parameter is largely from
 * \citet{fernandez-msc-1993, https://doi.org/10.1016/0168-583X(93)95827-R} Eq.
 * 32. For heavy particles, an empirical correction \f$ 1 + \exp(-(0.001 Z)^2)
 * \f$ is used to better match the data in \citet{attwood-scatteringmuons-2006,
 * https://doi.org/10.1016/j.nimb.2006.05.006} .
 * See also Bethe's re-derivation of Moliere scattering
 * \citep{bethe-msc-1953, https://doi.org/10.1103/PhysRev.89.1256}.
 *
 * See \cite{g4prm} section 8.5.
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
    inline CELER_FUNCTION
    WentzelHelper(ParticleTrackView const& particle,
                  MaterialView const& material,
                  AtomicNumber target_z,
                  NativeCRef<WentzelOKVIData> const& wentzel,
                  CoulombIds const& ids,
                  Energy cutoff);

    //! Get the target atomic number
    CELER_FUNCTION AtomicNumber atomic_number() const { return target_z_; }

    //! Get the Moliere screening coefficient
    CELER_FUNCTION real_type screening_coefficient() const
    {
        return screening_coefficient_;
    }

    //! Get the Mott factor (maximum, used for rejection)
    CELER_FUNCTION real_type mott_factor() const { return mott_factor_; }

    //! Get the multiplicative factor for the cross section
    CELER_FUNCTION real_type kin_factor() const { return kin_factor_; }

    //! Get the maximum scattering angle off of electrons
    CELER_FUNCTION real_type cos_thetamax_electron() const
    {
        return cos_thetamax_elec_;
    }

    //! Get the maximum scattering angle off of a nucleus
    CELER_FUNCTION real_type cos_thetamax_nuclear() const
    {
        return cos_thetamax_nuc_;
    }

    // Calculate the electron cross section for Coulomb scattering
    inline CELER_FUNCTION real_type
    calc_xs_electron(real_type cos_thetamin, real_type cos_thetamax) const;

    // Calculate the nuclear cross section for Coulomb scattering
    inline CELER_FUNCTION real_type
    calc_xs_nuclear(real_type cos_thetamin, real_type cos_thetamax) const;

  private:
    //// DATA ////

    AtomicNumber const target_z_;
    real_type screening_coefficient_;
    real_type kin_factor_;
    real_type mott_factor_;
    real_type cos_thetamax_elec_;
    real_type cos_thetamax_nuc_;

    //// HELPER FUNCTIONS ////

    // Calculate the screening coefficient R^2 for electrons
    static CELER_CONSTEXPR_FUNCTION MomentumSq screen_r_sq_elec();

    // Calculate the (cosine of) the maximum scattering angle off of electrons
    static inline CELER_FUNCTION real_type calc_cos_thetamax_electron(
        ParticleTrackView const&, CoulombIds const&, Energy, Mass);

    // Calculate the Moliere screening coefficient
    inline CELER_FUNCTION real_type calc_screening_coefficient(
        ParticleTrackView const&, CoulombIds const&) const;

    // Calculate the multiplicative factor for the cross section
    inline CELER_FUNCTION real_type calc_kin_factor(ParticleTrackView const&,
                                                    Mass) const;

    // Calculate the (cosine of) the maximum scattering angle off of a nucleus
    inline CELER_FUNCTION real_type calc_cos_thetamax_nuclear(
        ParticleTrackView const&,
        MaterialView const& material,
        NativeCRef<WentzelOKVIData> const& wentzel) const;

    // Calculate the common factor in the electron and nuclear cross section
    inline CELER_FUNCTION real_type calc_xs_factor(real_type cos_thetamin,
                                                   real_type cos_thetamax) const;
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
                             NativeCRef<WentzelOKVIData> const& wentzel,
                             CoulombIds const& ids,
                             Energy cutoff)
    : target_z_(target_z)
    , screening_coefficient_(this->calc_screening_coefficient(particle, ids)
                             * wentzel.params.screening_factor)
    , kin_factor_(this->calc_kin_factor(particle, wentzel.electron_mass))
    , mott_factor_(particle.particle_id() == ids.electron
                       ? 1 + real_type(2e-4) * ipow<2>(target_z_.get())
                       : 1)
    , cos_thetamax_elec_(this->calc_cos_thetamax_electron(
          particle, ids, cutoff, wentzel.electron_mass))
    , cos_thetamax_nuc_(
          this->calc_cos_thetamax_nuclear(particle, material, wentzel))
{
    CELER_EXPECT(screening_coefficient_ > 0);
    CELER_EXPECT(cos_thetamax_elec_ >= -1 && cos_thetamax_elec_ <= 1);
    CELER_EXPECT(cos_thetamax_nuc_ >= -1 && cos_thetamax_nuc_ <= 1);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the electron cross section for Coulomb scattering.
 */
CELER_FUNCTION real_type WentzelHelper::calc_xs_electron(
    real_type cos_thetamin, real_type cos_thetamax) const
{
    cos_thetamin = max(cos_thetamin, cos_thetamax_elec_);
    cos_thetamax = max(cos_thetamax, cos_thetamax_elec_);
    if (cos_thetamin <= cos_thetamax)
    {
        return 0;
    }
    return this->calc_xs_factor(cos_thetamin, cos_thetamax);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the nuclear cross section for Coulomb scattering.
 */
CELER_FUNCTION real_type WentzelHelper::calc_xs_nuclear(
    real_type cos_thetamin, real_type cos_thetamax) const
{
    return target_z_.get() * this->calc_xs_factor(cos_thetamin, cos_thetamax);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the common factor in the electron and nuclear cross section.
 */
CELER_FUNCTION real_type WentzelHelper::calc_xs_factor(
    real_type cos_thetamin, real_type cos_thetamax) const
{
    return kin_factor_ * mott_factor_ * (cos_thetamin - cos_thetamax)
           / ((1 - cos_thetamin + 2 * screening_coefficient_)
              * (1 - cos_thetamax + 2 * screening_coefficient_));
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the Moliere screening coefficient as in [PRM] eqn 8.51.
 *
 * \internal The \c screenZ in Geant4 is equal to twice the screening
 * coefficient.
 */
CELER_FUNCTION real_type WentzelHelper::calc_screening_coefficient(
    ParticleTrackView const& particle, CoulombIds const& ids) const
{
    // TODO: Reference for just proton correction?
    real_type correction = 1;
    real_type sq_cbrt_z = fastpow(real_type(target_z_.get()), real_type{2} / 3);
    if (target_z_.get() > 1)
    {
        // TODO: tau correction factor and "min" value are of unknown
        // provenance. The equation in Fernandez 1993 has factor=1, no special
        // casing for z=1, and no "min" for the correction
        real_type factor;
        real_type z_factor = 1;
        if (particle.particle_id() == ids.electron
            || particle.particle_id() == ids.positron)
        {
            // Electrons and positrons
            real_type tau = value_as<Energy>(particle.energy())
                            / value_as<Mass>(particle.mass());
            factor = std::sqrt(tau / (tau + sq_cbrt_z));
        }
        else
        {
            // Muons and hadrons
            factor = ipow<2>(value_as<Charge>(particle.charge()));
            z_factor += std::exp(-ipow<2>(target_z_.get()) * real_type(0.001));
        }
        correction = min(target_z_.get() * real_type{1.13},
                         real_type{1.13}
                             + real_type{3.76}
                                   * ipow<2>(target_z_.get()
                                             * constants::alpha_fine_structure)
                                   * factor / particle.beta_sq())
                     * z_factor;
    }

    return correction * sq_cbrt_z
           * value_as<MomentumSq>(this->screen_r_sq_elec())
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
CELER_CONSTEXPR_FUNCTION auto WentzelHelper::screen_r_sq_elec() -> MomentumSq
{
    //! Thomas-Fermi constant \f$ C_{TF} \f$
    constexpr Constant ctf{0.8853413770001135};

    return native_value_to<MomentumSq>(
        ipow<2>(constants::hbar_planck / (2 * ctf * constants::a0_bohr)));
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
CELER_FUNCTION real_type WentzelHelper::calc_kin_factor(
    ParticleTrackView const& particle, Mass electron_mass) const
{
    constexpr Constant twopirsq = 2 * constants::pi
                                  * ipow<2>(constants::r_electron);
    return twopirsq * target_z_.get()
           * ipow<2>(value_as<Mass>(electron_mass)
                     * value_as<Charge>(particle.charge()))
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
CELER_FUNCTION real_type
WentzelHelper::calc_cos_thetamax_electron(ParticleTrackView const& particle,
                                          CoulombIds const& ids,
                                          Energy cutoff,
                                          Mass electron_mass)
{
    real_type result = 0;
    real_type inc_energy = value_as<Energy>(particle.energy());
    real_type mass = value_as<Mass>(particle.mass());

    if (particle.particle_id() == ids.electron
        || particle.particle_id() == ids.positron)
    {
        // Electrons and positrons
        real_type max_energy = particle.particle_id() == ids.electron
                                   ? real_type{0.5} * inc_energy
                                   : inc_energy;
        real_type final_energy = inc_energy
                                 - min(value_as<Energy>(cutoff), max_energy);
        if (final_energy > 0)
        {
            real_type inc_ratio = 1 + 2 * mass / inc_energy;
            real_type final_ratio = 1 + 2 * mass / final_energy;
            result = clamp<real_type>(std::sqrt(inc_ratio / final_ratio), 0, 1);
        }
    }
    else
    {
        // Muons and hadrons
        real_type mass_ratio = value_as<Mass>(electron_mass) / mass;
        real_type tau = inc_energy / mass;
        real_type max_energy
            = 2 * value_as<Mass>(electron_mass) * tau * (tau + 2)
              / (1 + 2 * mass_ratio * (tau + 1) + ipow<2>(mass_ratio));
        result = -min(value_as<Energy>(cutoff), max_energy)
                 * value_as<Mass>(electron_mass)
                 / value_as<MomentumSq>(particle.momentum_sq());
    }
    CELER_ENSURE(result >= 0 && result <= 1);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the maximum scattering angle off the target nucleus.
 */
CELER_FUNCTION real_type WentzelHelper::calc_cos_thetamax_nuclear(
    ParticleTrackView const& particle,
    MaterialView const& material,
    NativeCRef<WentzelOKVIData> const& wentzel) const
{
    if (wentzel.params.is_combined)
    {
        CELER_ASSERT(material.material_id() < wentzel.inv_mass_cbrt_sq.size());
        return max(wentzel.params.costheta_limit,
                   1
                       - wentzel.params.a_sq_factor
                             * wentzel.inv_mass_cbrt_sq[material.material_id()]
                             / value_as<MomentumSq>(particle.momentum_sq()));
    }
    return wentzel.params.costheta_limit;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
