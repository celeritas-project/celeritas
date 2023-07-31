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
    WentzelDistribution(real_type inc_energy,
                        real_type inc_mass,
                        IsotopeView const& target,
                        WentzelElementData const& element_data,
                        real_type cutoff_energy,
                        bool is_electron,
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

    // Incident particle energy
    real_type const inc_energy_;

    // Incident particle mass
    real_type const inc_mass_;

    // Predicate for if the incident particle is an electron
    bool is_electron_;

    // Material cutoff energy for the incident particle
    real_type const cutoff_energy_;

    // Target isotope
    IsotopeView const& target_;

    // Mott coefficients for the target element
    WentzelElementData const& element_data_;

    //! Calculates the form factor from the given form coefficient and
    //! scattered polar angle
    inline CELER_FUNCTION real_type calculate_form_factor(real_type form_coeff,
                                                          real_type cos_t) const;
    //! Helper function for calculating the flat form factor
    inline CELER_FUNCTION real_type flat_form_factor(real_type x) const;

    //! Calculate incident momentum squared
    inline CELER_FUNCTION real_type inc_mom_sq() const;

    //! Calculate the nuclear form momentum scale
    inline CELER_FUNCTION real_type nuclear_form_momentum_scale() const;

    //! Calculate the screening coefficient R^2 for electrons
    CELER_CONSTEXPR_FUNCTION real_type screen_r_sq_elec() const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with state and data from WentzelInteractor
 */
CELER_FUNCTION
WentzelDistribution::WentzelDistribution(real_type inc_energy,
                                         real_type inc_mass,
                                         IsotopeView const& target,
                                         WentzelElementData const& element_data,
                                         real_type cutoff_energy,
                                         bool is_electron,
                                         WentzelRef const& data)
    : data_(data)
    , inc_energy_(inc_energy)
    , inc_mass_(inc_mass)
    , is_electron_(is_electron)
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
    UniformRealDistribution<real_type> uniform_sample;

    real_type screen_coeff = compute_screening_coefficient();
    real_type cos_t_max_elec = compute_max_electron_cos_t();

    // Parameters for scattering of a nucleus
    real_type form_factor_coeff = 0;
    real_type cos_t2 = -1;

    // Randomly choose if scattered off of electrons instead
    WentzelXsCalculator xsec(
        target_.atomic_number(), screen_coeff, cos_t_max_elec);
    real_type elec_ratio = xsec();
    if (uniform_sample(rng) < elec_ratio)
    {
        cos_t2 = max(cos_t2, cos_t_max_elec);
    }
    else
    {
        // Set nuclear form factor
        form_factor_coeff
            = inc_mom_sq()
              * fastpow(real_type(target_.atomic_mass_number().get()),
                        2 * real_type(0.27))
              / nuclear_form_momentum_scale();
    }

    // Sample scattering angle [Fern 92] where cos(theta) = 1 + 2*mu
    // For incident electrons / positrons, theta_min = 0 always
    real_type mu = real_type{0.5} * (1 - cos_t2);
    real_type xi = uniform_sample(rng);
    real_type cos_theta = clamp(
        1 + 2 * screen_coeff * mu * (1 - xi) / (screen_coeff - mu * xi),
        real_type{-1},
        real_type{1});

    // Calculate rejection for fake scattering
    // TODO: Reference?
    MottXsCalculator mott_xsec(element_data_, inc_energy_, inc_mass_);
    real_type form_factor = calculate_form_factor(form_factor_coeff, cos_theta);
    real_type g_rej = mott_xsec(cos_theta) * ipow<2>(form_factor);

    if (uniform_sample(rng) > g_rej)
    {
        return {0, 0, 1};
    }

    // Calculate scattered vector assuming azimuthal angle is isotropic
    real_type sin_theta = sqrt((1 - cos_theta) * (1 + cos_theta));
    real_type phi = 2 * celeritas::constants::pi * uniform_sample(rng);
    return {sin_theta * cos(phi), sin_theta * sin(phi), cos_theta};
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
CELER_FUNCTION real_type WentzelDistribution::calculate_form_factor(
    real_type formf, real_type cos_t) const
{
    switch (data_.form_factor_type)
    {
        case NuclearFormFactorType::flat: {
            // In units MeV
            const real_type ccoef = 0.00508;
            real_type x = sqrt(2 * inc_mom_sq() * (1 - cos_t)) * ccoef * 2;
            return flat_form_factor(x)
                   * flat_form_factor(
                       x * real_type{0.6}
                       * fastpow(value_as<Mass>(target_.nuclear_mass()),
                                 real_type{1} / 3));
        }
        break;
        case NuclearFormFactorType::exponential:
            return 1 / ipow<2>(1 + formf * (1 - cos_t));
        case NuclearFormFactorType::gaussian:
            return exp(-2 * formf * (1 - cos_t));
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
 * Helper function to calculate the incident momentum squared
 */
CELER_FUNCTION real_type WentzelDistribution::inc_mom_sq() const
{
    return inc_energy_ * (inc_energy_ + 2 * inc_mass_);
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
        real_type tau = inc_energy_ / inc_mass_;
        // TODO: Reference for this factor?
        real_type factor = sqrt(tau / (tau + sq_cbrt_z));
        real_type inv_beta_sq = 1 + ipow<2>(inc_mass_) / inc_mom_sq();

        correction = min(target_.atomic_number().get() * real_type{1.13},
                         real_type{1.13}
                             + real_type{3.76}
                                   * ipow<2>(target_.atomic_number().get()
                                             * constants::alpha_fine_structure)
                                   * inv_beta_sq * factor);
    }

    return correction * data_.screening_factor * screen_r_sq_elec() * sq_cbrt_z
           / inc_mom_sq();
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the (cosine of the) maximum polar angle that incident particle
 * can scatter off of the target's electrons.
 */
CELER_FUNCTION real_type WentzelDistribution::compute_max_electron_cos_t() const
{
    // TODO: Need to validate against Geant4 results
    real_type max_energy = is_electron_ ? real_type{0.5} * inc_energy_
                                        : inc_energy_;
    real_type final_energy = inc_energy_ - min(cutoff_energy_, max_energy);

    if (final_energy > 0)
    {
        real_type incident_ratio = 1 + 2 * inc_mass_ / inc_energy_;
        real_type final_ratio = 1 + 2 * inc_mass_ / final_energy;
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
 * Calculate the momentum scale for the nuclear form factor. This should
 * be a constant function but it has a special case for hydrogen.
 */
CELER_FUNCTION real_type WentzelDistribution::nuclear_form_momentum_scale() const
{
    // TODO: Geant has a different form momentum scale for hydrogen?
    if (target_.atomic_number().get() == 1)
    {
        return 1 / real_type{3.097e-6};
    }
    return native_value_to<MomentumSq>(
               12
               * ipow<2>(2 * constants::hbar_planck
                         / (real_type(1.27e-15) * units::meter)))
        .value();
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
}  // namespace celeritas
