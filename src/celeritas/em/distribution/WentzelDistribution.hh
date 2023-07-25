//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/distribution/MollerEnergyDistribution.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/em/data/WentzelData.hh"
#include "celeritas/em/xs/MottXsCalculator.hh"
#include "celeritas/em/xs/WentzelXsCalculator.hh"
#include "celeritas/mat/IsotopeView.hh"
#include "celeritas/random/distribution/UniformRealDistribution.hh"

/*
 * References:
 *      Fern - Fernandez-Varea 1992
 *      PRM - Geant4 Physics Reference Manual Release 11.1
 */
namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Helper class for \c WentzelInteractor .
 *
 * Samples the scattering direction for the Wentzel model.
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

    //! Target isotope's atomic number
    inline CELER_FUNCTION int target_Z() const;

    //! Incident momentum squared
    inline CELER_FUNCTION real_type inc_mom_sq() const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with state and data from WentzelInteractor
 *
 * TODO: Reference for factors?
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

    const real_type screen_coeff = compute_screening_coefficient();
    const real_type cos_t_max_elec = compute_max_electron_cos_t();

    // Parameters for scattering of a nucleus
    real_type form_factor_coeff = 0;
    real_type cos_t1 = 1;
    real_type cos_t2 = -1;

    // Randomly choose if scattered off of electrons instead
    const WentzelXsCalculator xsec(target_Z(), screen_coeff, cos_t_max_elec);
    const real_type elec_ratio = xsec();
    if (uniform_sample(rng) < elec_ratio)
    {
        // TODO: Can simplify this logic with
        //       -1 <= cos_t_max_elec <= 1
        cos_t1 = max(cos_t1, cos_t_max_elec);
        cos_t2 = max(cos_t2, cos_t_max_elec);
    }
    else
    {
        // TODO: Geant has a different form momentum scale for hydrogen?
        const real_type scale
            = (target_Z() == 1)
                  ? 1 / 3.097e-6
                  : value_as<MomentumSq>(data_.form_momentum_scale);

        // Set nuclear form factor
        form_factor_coeff
            = inc_mom_sq()
              * fastpow((real_type)target_.atomic_mass_number().get(), 2 * 0.27)
              / scale;
    }

    // Sample scattering angle [Fern 92] where cos(theta) = 1 + 2*mu
    // For incident electrons / positrons, theta_min = 0 always
    const real_type w1 = 1 - cos_t1 + 2 * screen_coeff;
    const real_type w2 = 1 - cos_t2 + 2 * screen_coeff;
    const real_type cos_theta = clamp(
        1 + 2 * screen_coeff - w1 * w2 / (w1 + uniform_sample(rng) * (w2 - w1)),
        real_type{-1},
        real_type{1});

    // Calculate rejection for fake scattering
    // TODO: Reference?
    MottXsCalculator mott_xsec(element_data_, inc_energy_, inc_mass_);
    const real_type form_factor
        = calculate_form_factor(form_factor_coeff, cos_theta);
    const real_type g_rej = mott_xsec(cos_theta) * ipow<2>(form_factor);

    if (uniform_sample(rng) > g_rej)
    {
        return {0, 0, 1};
    }

    // Calculate scattered vector assuming azimuthal angle is isotropic
    const real_type sin_theta = sqrt((1 - cos_theta) * (1 + cos_theta));
    const real_type phi = 2 * celeritas::constants::pi * uniform_sample(rng);
    return {sin_theta * cos(phi), sin_theta * sin(phi), cos_theta};
}

//---------------------------------------------------------------------------//
/*!
 * Calculates the form factor based on the form factor model.
 * TODO: Reference?
 */
CELER_FUNCTION real_type WentzelDistribution::calculate_form_factor(
    real_type formf, real_type cos_t) const
{
    switch (data_.form_factor_type)
    {
        case NuclearFormFactorType::Flat: {
            // In units MeV
            const real_type ccoef = 0.00508;
            const real_type x = sqrt(2 * inc_mom_sq() * (1 - cos_t)) * ccoef
                                * 2;
            return flat_form_factor(x)
                   * flat_form_factor(
                       x * 0.6
                       * fastpow(value_as<Mass>(target_.nuclear_mass()),
                                 static_cast<real_type>(1) / 3));
        }
        break;
        case NuclearFormFactorType::Exponential:
            return 1 / ipow<2>(1 + formf * (1 - cos_t));
        case NuclearFormFactorType::Gaussian:
            return exp(-2 * formf * (1 - cos_t));
        default:
            return 1;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Flat form factor
 * TODO: Reference?
 */
CELER_FUNCTION real_type WentzelDistribution::flat_form_factor(real_type x) const
{
    return 3 * (sin(x) - x * cos(x)) / ipow<3>(x);
}

//---------------------------------------------------------------------------//
/*!
 * Atomic number of the target isotope
 */
CELER_FUNCTION int WentzelDistribution::target_Z() const
{
    return target_.atomic_number().get();
}

//---------------------------------------------------------------------------//
/*!
 * Helper function to calculate the incident momentum squared.
 */
CELER_FUNCTION real_type WentzelDistribution::inc_mom_sq() const
{
    return inc_energy_ * (inc_energy_ + 2 * inc_mass_);
}

//---------------------------------------------------------------------------//
/*!
 * Calculates the Moilere screening coefficient.
 */
CELER_FUNCTION real_type WentzelDistribution::compute_screening_coefficient() const
{
    // TODO: Reference for just proton correction?
    real_type correction = 1;
    const real_type sq_cbrt_z = fastpow(static_cast<real_type>(target_Z()),
                                        static_cast<real_type>(2) / 3);
    if (target_Z() > 1)
    {
        const real_type tau = inc_energy_ / inc_mass_;
        // TODO: Reference for this factor?
        const real_type factor = sqrt(tau / (tau + sq_cbrt_z));
        const real_type inv_beta_sq = 1 + ipow<2>(inc_mass_) / inc_mom_sq();

        correction = min(
            target_Z() * 1.13,
            1.13
                + 3.76 * ipow<2>(target_Z() * constants::alpha_fine_structure)
                      * inv_beta_sq * factor);
    }

    return correction * value_as<MomentumSq>(data_.screen_r_sq_elec)
           * sq_cbrt_z / inc_mom_sq();
}

//---------------------------------------------------------------------------//
/*!
 * Calculates the (cosine of the) maximum polar angle that incident particle
 * can scatter off of the target's electrons.
 */
CELER_FUNCTION real_type WentzelDistribution::compute_max_electron_cos_t() const
{
    // TODO: Need to validate against Geant4 results
    const real_type max_energy = is_electron_ ? inc_energy_ / 2 : inc_energy_;
    const real_type transferred_energy = min(cutoff_energy_, max_energy);

    // Incident and transferred energy ratios
    const real_type r1 = transferred_energy
                         / (transferred_energy + 2 * inc_mass_);
    const real_type r2 = inc_energy_ / (inc_energy_ + 2 * inc_mass_);
    const real_type ctm = sqrt(r1 / r2);
    return clamp(ctm, real_type{0}, real_type{1});
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
