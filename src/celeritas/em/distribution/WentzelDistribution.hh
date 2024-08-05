//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
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
#include "celeritas/em/interactor/detail/PhysicsConstants.hh"
#include "celeritas/em/xs/MottRatioCalculator.hh"
#include "celeritas/em/xs/WentzelHelper.hh"
#include "celeritas/mat/IsotopeView.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/random/distribution/BernoulliDistribution.hh"
#include "celeritas/random/distribution/GenerateCanonical.hh"
#include "celeritas/random/distribution/RejectionSampler.hh"
#include "celeritas/random/distribution/UniformRealDistribution.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Helper class for \c CoulombScatteringInteractor .
 *
 * Samples the polar scattering angle for the Wentzel Coulomb scattering model.
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
    // Construct with state and model data
    inline CELER_FUNCTION
    WentzelDistribution(NativeCRef<WentzelOKVIData> const& wentzel,
                        WentzelHelper const& helper,
                        ParticleTrackView const& particle,
                        IsotopeView const& target,
                        ElementId el_id,
                        real_type cos_thetamin,
                        real_type cos_thetamax);

    // Sample the polar scattering angle
    template<class Engine>
    inline CELER_FUNCTION real_type operator()(Engine& rng) const;

  private:
    //// DATA ////

    // Shared Coulomb scattering data
    NativeCRef<WentzelOKVIData> const& wentzel_;

    // Helper for calculating xs ratio and other quantities
    WentzelHelper const& helper_;

    // Incident particle
    ParticleTrackView const& particle_;

    // Target isotope
    IsotopeView const& target_;

    // Target element
    ElementId el_id_;

    // Cosine of the minimum scattering angle
    real_type cos_thetamin_;

    // Cosine of the maximum scattering angle
    real_type cos_thetamax_;

    //// HELPER FUNCTIONS ////

    // Calculates the form factor from the scattered polar angle
    inline CELER_FUNCTION real_type calculate_form_factor(real_type cos_t) const;

    // Calculate the nuclear form momentum scale
    inline CELER_FUNCTION real_type nuclear_form_momentum_scale() const;

    // Calculate the squared momentum transfer to the target
    inline CELER_FUNCTION real_type mom_transfer_sq(real_type cos_t) const;

    // Momentum prefactor used in exponential and gaussian form factors
    inline CELER_FUNCTION real_type nuclear_form_prefactor() const;

    // Sample the scattered polar angle
    template<class Engine>
    inline CELER_FUNCTION real_type sample_costheta(real_type cos_thetamin,
                                                    real_type cos_thetamax,
                                                    Engine& rng) const;

    // Helper function for calculating the flat form factor
    inline static CELER_FUNCTION real_type flat_form_factor(real_type x);

    // Momentum coefficient used in the flat nuclear form factor model
    static CELER_CONSTEXPR_FUNCTION real_type flat_coeff();
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with state and model data.
 */
CELER_FUNCTION
WentzelDistribution::WentzelDistribution(
    NativeCRef<WentzelOKVIData> const& wentzel,
    WentzelHelper const& helper,
    ParticleTrackView const& particle,
    IsotopeView const& target,
    ElementId el_id,
    real_type cos_thetamin,
    real_type cos_thetamax)
    : wentzel_(wentzel)
    , helper_(helper)
    , particle_(particle)
    , target_(target)
    , el_id_(el_id)
    , cos_thetamin_(cos_thetamin)
    , cos_thetamax_(cos_thetamax)
{
    CELER_EXPECT(el_id_ < wentzel_.elem_data.size());
    CELER_EXPECT(cos_thetamin_ >= -1 && cos_thetamin_ <= 1);
    CELER_EXPECT(cos_thetamax_ >= -1 && cos_thetamax_ <= 1);
    CELER_EXPECT(cos_thetamax_ <= cos_thetamin_);
}

//---------------------------------------------------------------------------//
/*!
 * Sample the polar scattered angle of the incident particle.
 */
template<class Engine>
CELER_FUNCTION real_type WentzelDistribution::operator()(Engine& rng) const
{
    real_type cos_theta = 1;
    if (BernoulliDistribution(
            helper_.calc_xs_electron(cos_thetamin_, cos_thetamax_),
            helper_.calc_xs_nuclear(cos_thetamin_, cos_thetamax_))(rng))
    {
        // Scattered off of electrons
        real_type const cos_thetamax_elec = helper_.cos_thetamax_electron();
        real_type cos_thetamin = max(cos_thetamin_, cos_thetamax_elec);
        real_type cos_thetamax = max(cos_thetamax_, cos_thetamax_elec);
        CELER_ASSERT(cos_thetamin > cos_thetamax);
        cos_theta = this->sample_costheta(cos_thetamin, cos_thetamax, rng);
    }
    else
    {
        // Scattered off of nucleus
        cos_theta = this->sample_costheta(cos_thetamin_, cos_thetamax_, rng);

        // Calculate rejection for fake scattering
        // TODO: Reference?
        MottRatioCalculator mott_xsec(wentzel_.elem_data[el_id_],
                                      std::sqrt(particle_.beta_sq()));
        real_type xs = mott_xsec(cos_theta)
                       * ipow<2>(this->calculate_form_factor(cos_theta));
        if (RejectionSampler(xs, helper_.mott_factor())(rng))
        {
            // Reject scattering event: no change in direction
            cos_theta = 1;
        }
    }
    return cos_theta;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the form factor based on the form factor model.
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
    real_type mt_sq = this->mom_transfer_sq(cos_t);
    switch (wentzel_.params.form_factor_type)
    {
        case NuclearFormFactorType::none:
            return 1;
        case NuclearFormFactorType::flat: {
            real_type x1 = this->flat_coeff() * std::sqrt(mt_sq);
            real_type x0 = real_type{0.6} * x1
                           * fastpow(value_as<Mass>(target_.nuclear_mass()),
                                     real_type{1} / 3);
            return this->flat_form_factor(x0) * this->flat_form_factor(x1);
        }
        case NuclearFormFactorType::exponential:
            return 1 / ipow<2>(1 + this->nuclear_form_prefactor() * mt_sq);
        case NuclearFormFactorType::gaussian:
            return std::exp(-2 * this->nuclear_form_prefactor() * mt_sq);
        default:
            CELER_ASSERT_UNREACHABLE();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the flat form factor.
 *
 * See [LR15] eqn 2.265.
 */
CELER_FUNCTION real_type WentzelDistribution::flat_form_factor(real_type x)
{
    return 3 * (std::sin(x) - x * std::cos(x)) / ipow<3>(x);
}

//---------------------------------------------------------------------------//
/*!
 * Momentum coefficient used in the flat model for the nuclear form factors.
 *
 * This is the ratio of \f$ r_1 / \hbar \f$ where \f$ r_1 \f$ is defined in
 * eqn 2.265 of [LR15].
 */
CELER_CONSTEXPR_FUNCTION real_type WentzelDistribution::flat_coeff()
{
    return native_value_to<units::MevMomentum>(2 * units::femtometer
                                               / constants::hbar_planck)
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
 * Get the constant prefactor of the squared momentum transfer.
 */
CELER_FUNCTION real_type WentzelDistribution::nuclear_form_prefactor() const
{
    CELER_EXPECT(target_.isotope_id() < wentzel_.nuclear_form_prefactor.size());
    return wentzel_.nuclear_form_prefactor[target_.isotope_id()];
}

//---------------------------------------------------------------------------//
/*!
 * Sample the scattering polar angle of the incident particle.
 *
 * The probability is given in [Fern] eqn 88 and is nomalized on the interval
 * \f$ cos\theta \in [\cos\theta_{min}, \cos\theta_{max}] \f$. The sampling
 * function for \f$ \mu = \frac{1}{2}(1 - \cos\theta) \f$ is
 * \f[
   \mu = \mu_1 + \frac{(A + \mu_1) \xi (\mu_2 - \mu_1)}{A + \mu_2 - \xi (\mu_2
   - \mu_1)},
 * \f]
 * where \f$ \mu_1 = \frac{1}{2}(1 - \cos\theta_{min}) \f$, \f$ \mu_2 =
 * \frac{1}{2}(1 - \cos\theta_{max}) \f$, \f$ A \f$ is the screening
 * coefficient, and \f$ \xi \sim U(0,1) \f$.
 */
template<class Engine>
CELER_FUNCTION real_type WentzelDistribution::sample_costheta(
    real_type cos_thetamin, real_type cos_thetamax, Engine& rng) const
{
    // Sample scattering angle [Fern] eqn 92, where cos(theta) = 1 - 2*mu
    real_type const mu1 = real_type{0.5} * (1 - cos_thetamin);
    real_type const mu2 = real_type{0.5} * (1 - cos_thetamax);
    real_type const w = generate_canonical(rng) * (mu2 - mu1);
    real_type const sc = helper_.screening_coefficient();

    real_type result = 1 - 2 * mu1 - 2 * (sc + mu1) * w / (sc + mu2 - w);
    CELER_ENSURE(result >= -1 && result <= 1);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
