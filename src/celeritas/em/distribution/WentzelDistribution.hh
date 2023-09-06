//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/distribution/WentzelDistribution.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/em/data/WentzelData.hh"
#include "celeritas/em/interactor/detail/PhysicsConstants.hh"
#include "celeritas/em/xs/MottRatioCalculator.hh"
#include "celeritas/em/xs/WentzelRatioCalculator.hh"
#include "celeritas/mat/IsotopeView.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/random/distribution/BernoulliDistribution.hh"
#include "celeritas/random/distribution/GenerateCanonical.hh"
#include "celeritas/random/distribution/UniformRealDistribution.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Helper class for \c WentzelInteractor .
 *
 * Samples the polar scattering angle for the Wentzel model.
 *
 * References:
 * [Fern] J.M. Fernandez-Varea, R. Mayol and F. Salvat. On the theory
 *        and simulation of multiple elastic scattering of electrons. Nucl.
 *        Instrum. and Method. in Phys. Research B, 73:447-473, Apr 1993.
 *        doi:10.1016/0168-583X(93)95827-R
 * [LR15] C. Leroy and P.G. Rancoita. Principles of Radiation Interaction in
 *        Matter and Detection. World Scientific (Singapore), 4rd edition,
 *        2015.
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

    //! Sample the polar scattering angle
    template<class Engine>
    inline CELER_FUNCTION real_type operator()(Engine& rng) const;

  private:
    //// DATA ////

    // Shared WentzelModel data
    WentzelRef const& data_;

    // Incident particle
    ParticleTrackView const& particle_;

    // Target isotope
    IsotopeView const& target_;

    // Mott coefficients for the target element
    WentzelElementData const& element_data_;

    // Ratio of elecron to total cross sections for the Wentzel model
    WentzelRatioCalculator const wentzel_elec_ratio;

    //! Calculates the form factor from the scattered polar angle
    inline CELER_FUNCTION real_type calculate_form_factor(real_type cos_t) const;

    //! Helper function for calculating the flat form factor
    inline CELER_FUNCTION real_type flat_form_factor(real_type x) const;

    //! Calculate the nuclear form momentum scale
    inline CELER_FUNCTION real_type nuclear_form_momentum_scale() const;

    //! Calculate the squared momentum transfer to the target
    inline CELER_FUNCTION real_type mom_transfer_sq(real_type cos_t) const;

    //! Momentum coefficient used in the flat nuclear form factor model
    CELER_CONSTEXPR_FUNCTION real_type flat_coeff() const;

    //! Momentum prefactor used in exponential and gaussian form factors
    inline CELER_FUNCTION real_type nuclear_form_prefactor() const;

    //! Samples the scattered polar angle based on the maximum scattering
    //! angle and screening coefficient
    template<class Engine>
    inline CELER_FUNCTION real_type sample_cos_t(real_type cos_t_max,
                                                 Engine& rng) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with state and data from WentzelInteractor.
 */
CELER_FUNCTION
WentzelDistribution::WentzelDistribution(ParticleTrackView const& particle,
                                         IsotopeView const& target,
                                         WentzelElementData const& element_data,
                                         real_type cutoff_energy,
                                         WentzelRef const& data)
    : data_(data)
    , particle_(particle)
    , target_(target)
    , element_data_(element_data)
    , wentzel_elec_ratio(particle, target.atomic_number(), data, cutoff_energy)
{
}

//---------------------------------------------------------------------------//
/*!
 * Samples the polar scattered angle of the incident particle.
 */
template<class Engine>
CELER_FUNCTION real_type WentzelDistribution::operator()(Engine& rng) const
{
    real_type cos_theta = 1;
    if (BernoulliDistribution(wentzel_elec_ratio())(rng))
    {
        // Scattered off of electrons
        cos_theta = sample_cos_t(wentzel_elec_ratio.cos_t_max_elec(), rng);
    }
    else
    {
        // Scattered off of nucleus
        cos_theta = sample_cos_t(-1, rng);

        // Calculate rejection for fake scattering
        // TODO: Reference?
        real_type mott_coeff
            = 1 + real_type(1e-4) * ipow<2>(target_.atomic_number().get());
        MottRatioCalculator mott_xsec(element_data_,
                                      std::sqrt(particle_.beta_sq()));
        real_type g_rej = mott_xsec(cos_theta)
                          * ipow<2>(calculate_form_factor(cos_theta))
                          / mott_coeff;

        if (!BernoulliDistribution(g_rej)(rng))
        {
            // Reject scattering event: no change in direction
            cos_theta = 1;
        }
    }

    return cos_theta;
}

//---------------------------------------------------------------------------//
/*!
 * Calculates the form factor based on the form factor model.
 *
 * The models are described in [LR15] section 2.4.2.1 and parameterize the
 * charge distribution inside a nucleus. The same models are used as in
 * Geant4, and are:
 *      Exponential: [LR15] eqn 2.262
 *      Gaussian: [LR15] eqn 2.264
 *      Flat (uniform-uniform folded): [LR15] 2.265
 */
CELER_FUNCTION real_type
WentzelDistribution::calculate_form_factor(real_type cos_t) const
{
    switch (data_.form_factor_type)
    {
        case NuclearFormFactorType::flat: {
            real_type x1 = flat_coeff() * std::sqrt(mom_transfer_sq(cos_t));
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
            return std::exp(-2 * nuclear_form_prefactor()
                            * mom_transfer_sq(cos_t));
        default:
            return 1;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Helper function for calculating the flat form factors, see [LR15] eqn
 * 2.265.
 */
CELER_FUNCTION real_type WentzelDistribution::flat_form_factor(real_type x) const
{
    return 3 * (std::sin(x) - x * std::cos(x)) / ipow<3>(x);
}

//---------------------------------------------------------------------------//
/*!
 * Momentum coefficient used in the flat model for the nuclear form factors.
 * This is the ratio of \f$ r_1 / \hbar \f$ where \f$ r_1 \f$ is defined in
 * eqn 2.265 of [LR15].
 */
CELER_CONSTEXPR_FUNCTION real_type WentzelDistribution::flat_coeff() const
{
    return native_value_to<units::MevMomentum>(2 * units::femtometer
                                               / constants::hbar_planck)
        .value();
}

//---------------------------------------------------------------------------//
/*!
 * Calculates the constant prefactors of the squared momentum transfer used
 * in the exponential and Guassian nuclear form models, see eqns 2.262-2.264
 * of [LR15].
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
    // inverse which has units of MomentumSq, then invert afterwards
    constexpr real_type ratio
        = 1
          / native_value_to<MomentumSq>(
                12
                * ipow<2>(constants::hbar_planck
                          / (real_type(1.27) * units::femtometer)))
                .value();
    return ratio
           * fastpow(real_type(target_.atomic_mass_number().get()),
                     2 * real_type(0.27));
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
 */
template<class Engine>
CELER_FUNCTION real_type WentzelDistribution::sample_cos_t(real_type cos_t_max,
                                                           Engine& rng) const
{
    // Sample scattering angle [Fern] eqn 92, where cos(theta) = 1 - 2*mu
    // For incident electrons / positrons, theta_min = 0 always
    real_type const mu = real_type{0.5} * (1 - cos_t_max);
    real_type const xi = generate_canonical(rng);
    real_type const sc = wentzel_elec_ratio.screening_coefficient();

    return clamp(1 + 2 * sc * mu * (1 - xi) / (sc - mu * xi),
                 real_type{-1},
                 real_type{1});
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
