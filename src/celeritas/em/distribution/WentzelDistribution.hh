//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/distribution/WentzelDistribution.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/em/data/WentzelData.hh"
#include "celeritas/em/interactor/detail/PhysicsConstants.hh"
#include "celeritas/em/xs/MottXsCalculator.hh"
#include "celeritas/em/xs/WentzelXsCalculator.hh"
#include "celeritas/mat/IsotopeView.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/random/distribution/BernoulliDistribution.hh"
#include "celeritas/random/distribution/UniformRealDistribution.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Helper class for \c WentzelInteractor .
 *
 * Samples the scattering direction for the Wentzel model.
 *
 * References:
 * [Fern] J.M. Fernandez-Varea, R. Mayol and F. Salvat. On the theory
 *        and simulation of multiple elastic scattering of electrons. Nucl.
 *        Instrum. and Method. in Phys. Research B, 73:447-473, Apr 1993.
 *        doi:10.1016/0168-583X(93)95827-R
 * [LR11] C. Leroy and P.G. Rancoita. Principles of Radiation Interaction in
 *        Matter and Detection. World Scientific (Singapore), 3rd edition,
 *        2011.
 * [PRM]  Geant4 Physics Reference Manual (Release 11.1) sections 8.2 and 8.5
 */
class WentzelDistribution
{
  public:
    //!@{
    //! \name Type aliases
    using MomentumSq = units::MevMomentumSq;
    using Energy = units::MevEnergy;
    using Mass = units::MevMass;
    //!@}

  public:
    //! Construct with state and date from WentzelInteractor
    inline CELER_FUNCTION
    WentzelDistribution(ParticleTrackView const& particle,
                        IsotopeView const& target,
                        WentzelElementData const& element_data,
                        real_type cutoff_energy,
                        WentzelRef const& data);

    //! Sample the scattering direction
    template<class Engine>
    inline CELER_FUNCTION Real3 operator()(Engine& rng) const;

    //! Calculates the Moliere screening coefficient for the given target
    inline CELER_FUNCTION real_type compute_screening_coefficient() const;

    //! Calculates the cosine of the maximum scattering angle
    inline CELER_FUNCTION real_type compute_max_electron_cos_t() const;

  private:
    //// DATA ////

    // Shared WentzelModel data
    WentzelRef const& data_;

    // Incident particle
    ParticleTrackView const& particle_;

    // Material cutoff energy for the incident particle
    real_type const cutoff_energy_;

    // Target isotope
    IsotopeView const& target_;

    // Mott coefficients for the target element
    WentzelElementData const& element_data_;

    //! Calculates the form factor from the scattered polar angle
    inline CELER_FUNCTION real_type calculate_form_factor(real_type cos_t) const;

    //! Helper function for calculating the flat form factor
    inline CELER_FUNCTION real_type flat_form_factor(real_type x) const;

    //! Calculate the nuclear form momentum scale
    inline CELER_FUNCTION real_type nuclear_form_momentum_scale() const;

    //! Calculate the screening coefficient R^2 for electrons
    CELER_CONSTEXPR_FUNCTION real_type screen_r_sq_elec() const;

    //! Calculate the squared momentum transfer to the target
    inline CELER_FUNCTION real_type mom_transfer_sq(real_type cos_t) const;

    //! Momentum coefficient used in the flat nuclear form factor model
    CELER_CONSTEXPR_FUNCTION real_type flat_coeff() const;

    //! Momentum prefactor used in exponential and gaussian form factors
    inline CELER_FUNCTION real_type nuclear_form_prefactor() const;

    //! Samples the scattered polar angle based on the maximum scattering
    //! angle and screening coefficient.
    template<class Engine>
    inline CELER_FUNCTION real_type sample_cos_t(real_type screen_coeff,
                                                 real_type cos_t_max,
                                                 Engine& rng) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with state and data from WentzelInteractor
 */
CELER_FUNCTION
WentzelDistribution::WentzelDistribution(ParticleTrackView const& particle,
                                         IsotopeView const& target,
                                         WentzelElementData const& element_data,
                                         real_type cutoff_energy,
                                         WentzelRef const& data)
    : data_(data)
    , particle_(particle)
    , cutoff_energy_(cutoff_energy)
    , target_(target)
    , element_data_(element_data)
{
}

//---------------------------------------------------------------------------//
/*!
 * Samples the final direction of the interaction. This direction is in
 * the frame where the incident particle's momentum is oriented along the
 * z-axis, so it's final direction in the lab frame will need to be rotated.
 */
template<class Engine>
CELER_FUNCTION Real3 WentzelDistribution::operator()(Engine& rng) const
{
    real_type screen_coeff = compute_screening_coefficient();
    real_type cos_t_max_elec = compute_max_electron_cos_t();

    // Randomly choose if scattered off of electrons instead
    WentzelXsCalculator calc_electron_prob(
        target_.atomic_number(), screen_coeff, cos_t_max_elec);

    real_type cos_theta = 1;
    if (BernoulliDistribution(calc_electron_prob())(rng))
    {
        // Scattered off of electrons
        cos_theta = sample_cos_t(screen_coeff, cos_t_max_elec, rng);
    }
    else
    {
        // Scattered off of nucleus
        cos_theta = sample_cos_t(screen_coeff, -1, rng);

        // Calculate rejection for fake scattering
        // TODO: Reference?
        MottXsCalculator mott_xsec(element_data_, sqrt(particle_.beta_sq()));
        real_type g_rej = mott_xsec(cos_theta)
                          * ipow<2>(calculate_form_factor(cos_theta));

        if (g_rej <= 1 && !BernoulliDistribution(g_rej)(rng))
        {
            return {0, 0, 1};
        }
    }

    // Calculate scattered vector assuming azimuthal angle is isotropic
    UniformRealDistribution<real_type> sample_phi(0, 2 * constants::pi);
    return from_spherical(cos_theta, sample_phi(rng));
}

//---------------------------------------------------------------------------//
/*!
 * Calculates the form factor based on the form factor model.
 *
 * The models are described in [LR11] section 2.4.2.1 and parameterize the
 * charge distribution inside a nucleus. The same models are used as in
 * Geant4, and are:
 *      Exponential: [LR11] eqn 2.262
 *      Gaussian: [LR11] eqn 2.264
 *      Flat (uniform-uniform folded): [LR11] 2.265
 */
CELER_FUNCTION real_type
WentzelDistribution::calculate_form_factor(real_type cos_t) const
{
    switch (data_.form_factor_type)
    {
        case NuclearFormFactorType::flat: {
            real_type x1 = flat_coeff() * sqrt(mom_transfer_sq(cos_t));
            real_type x0 = real_type{0.6} * x1
                           * fastpow(value_as<Mass>(target_.nuclear_mass()),
                                     real_type{1} / 3);
            return flat_form_factor(x0) * flat_form_factor(x1);
        }
        case NuclearFormFactorType::exponential:
            return 1
                   / ipow<2>(
                       1 + nuclear_form_prefactor() * mom_transfer_sq(cos_t));
        case NuclearFormFactorType::gaussian:
            return exp(-2 * nuclear_form_prefactor() * mom_transfer_sq(cos_t));
        default:
            return 1;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Helper function for calculating the flat form factors, see [LR16] eqn
 * 2.265.
 */
CELER_FUNCTION real_type WentzelDistribution::flat_form_factor(real_type x) const
{
    return 3 * (sin(x) - x * cos(x)) / ipow<3>(x);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the Moilere screening coefficient.
 */
CELER_FUNCTION real_type WentzelDistribution::compute_screening_coefficient() const
{
    // TODO: Reference for just proton correction?
    real_type correction = 1;
    real_type sq_cbrt_z
        = fastpow(real_type(target_.atomic_number().get()), real_type{2} / 3);
    if (target_.atomic_number().get() > 1)
    {
        real_type tau = value_as<Energy>(particle_.energy())
                        / value_as<Mass>(particle_.mass());
        // TODO: Reference for this factor?
        real_type factor = sqrt(tau / (tau + sq_cbrt_z));

        correction = min(target_.atomic_number().get() * real_type{1.13},
                         real_type{1.13}
                             + real_type{3.76}
                                   * ipow<2>(target_.atomic_number().get()
                                             * constants::alpha_fine_structure)
                                   * factor / particle_.beta_sq());
    }

    return correction * data_.screening_factor * screen_r_sq_elec() * sq_cbrt_z
           / value_as<MomentumSq>(particle_.momentum_sq());
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the (cosine of the) maximum polar angle that incident particle
 * can scatter off of the target's electrons.
 */
CELER_FUNCTION real_type WentzelDistribution::compute_max_electron_cos_t() const
{
    real_type inc_energy = value_as<Energy>(particle_.energy());
    real_type mass = value_as<Mass>(particle_.mass());

    // TODO: Need to validate against Geant4 results
    real_type max_energy = (particle_.particle_id() == data_.ids.electron)
                               ? real_type{0.5} * inc_energy
                               : inc_energy;
    real_type final_energy = inc_energy - min(cutoff_energy_, max_energy);

    if (final_energy > 0)
    {
        real_type incident_ratio = 1 + 2 * mass / inc_energy;
        real_type final_ratio = 1 + 2 * mass / final_energy;
        real_type cos_t_max = sqrt(incident_ratio / final_ratio);

        return clamp(cos_t_max, real_type{0}, real_type{1});
    }
    else
    {
        return 0;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Momentum coefficient used in the flat model for the nuclear form factors.
 * This is the ratio of \f$ r_1 / \hbar \f$ where \f$ r_1 \f$ is defined in
 * eqn 2.265 of [LR11].
 */
CELER_CONSTEXPR_FUNCTION real_type WentzelDistribution::flat_coeff() const
{
    return native_value_to<units::MevMomentum>(real_type(2e-15) * units::meter
                                               / constants::hbar_planck)
        .value();
}

//---------------------------------------------------------------------------//
/*!
 * Calculates the constant prefactors of the squared momentum transfer used
 * in the exponential and Guassian nuclear form models, see eqns 2.262-2.264
 * of [LR11].
 *
 * Specifically, it calculates (r_n/hbar)^2 / 12. A special case is inherited
 * from Geant for hydrogen targets.
 */
CELER_FUNCTION real_type WentzelDistribution::nuclear_form_prefactor() const
{
    // TODO: Geant has a different prefactor for hydrogen?
    if (target_.atomic_number().get() == 1)
    {
        return real_type{1.5485e-6};
    }

    // The ratio has units of (MeV/c)^-2, so it's easier to convert the
    // inverse which has units of MomentumSq, then invert afterwards.
    constexpr real_type ratio
        = 1
          / native_value_to<MomentumSq>(
                12
                * ipow<2>(constants::hbar_planck
                          / (real_type(1.27e-15) * units::meter)))
                .value();
    return ratio
           * fastpow(real_type(target_.atomic_mass_number().get()),
                     2 * real_type(0.27));
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the screening R^2 coefficient for incident electrons. This is
 * the constant prefactor of [PRM] eqn 8.51
 */
CELER_CONSTEXPR_FUNCTION real_type WentzelDistribution::screen_r_sq_elec() const
{
    //! Thomas-Fermi constant C_TF
    //! \f$ \frac{1}{2}\left(\frac{3\pi}{4}\right)^{2/3} \f$
    constexpr real_type ctf = 0.8853413770001135;

    return native_value_to<MomentumSq>(
               ipow<2>(constants::hbar_planck / (2 * ctf * constants::a0_bohr)))
        .value();
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the squared momentum transfer to the target for the given
 * deflection angle \f$\cos\theta\f$ of the incident particle.
 *
 * This is an approximation for small recoil energies.
 */
CELER_FUNCTION real_type WentzelDistribution::mom_transfer_sq(real_type cos_t) const
{
    return 2 * value_as<MomentumSq>(particle_.momentum_sq()) * (1 - cos_t);
}

//---------------------------------------------------------------------------//
/*!
 * Randomly samples the scattering polar angle of the incident particle
 * based on the maximum scattering angle. The probability is given in [Fern]
 * eqn 88 and is nomalized on the interval
 * \f$ cos\theta \in [1, \cos\theta_{max}] \f$.
 *
 * The screening coefficient is also supplied since it's calculation requires
 * branching and it's already calculated for the Wentzel cross sections.
 */
template<class Engine>
CELER_FUNCTION real_type WentzelDistribution::sample_cos_t(
    real_type screen_coeff, real_type cos_t_max, Engine& rng) const
{
    UniformRealDistribution<real_type> uniform_sample;

    // For incident electrons / positrons, theta_min = 0 always
    real_type mu = real_type{0.5} * (1 - cos_t_max);
    real_type xi = uniform_sample(rng);

    return clamp(
        1 + 2 * screen_coeff * mu * (1 - xi) / (screen_coeff - mu * xi),
        real_type{-1},
        real_type{1});
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
