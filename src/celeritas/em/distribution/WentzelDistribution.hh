//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/distribution/WentzelDistribution.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/random/distribution/BernoulliDistribution.hh"
#include "corecel/random/distribution/RejectionSampler.hh"
#include "celeritas/em/interactor/detail/PhysicsConstants.hh"
#include "celeritas/em/xs/MottRatioCalculator.hh"
#include "celeritas/em/xs/NuclearFormFactors.hh"
#include "celeritas/em/xs/WentzelHelper.hh"
#include "celeritas/mat/IsotopeView.hh"
#include "celeritas/phys/ParticleTrackView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Sample the polar scattering angle cosine for Wentzel Coulomb scattering.
 *
 * This chooses between sampling scattering off an electron or nucleus based on
 * the relative cross sections. Electron scattering angle imposes a maximum
 * scattering angle, and nuclear
 * sattering rejects an angular change based on the Mott cross section (see
 * MottRatioCalculator). Nuclear scattering depends on the electronic and
 * nuclear cross sections (calculated by \c WentzelHelper) and nuclear form
 * factors (see \c ExpNuclearFormFactor, \c GaussianNuclearFormFactor, and \c
 * UUNuclearFormFactor ) that describe an approximate spatial charge
 * distribution of the nucleus. The class is used by \c
 * CoulombScatteringInteractor .
 *
 * The polar angle distribution is given in
 * \citet{fernandez-msc-1993, https://doi.org/10.1016/0168-583X(93)95827-R}
 * Eq. 88 and is normalized on the interval
 * \f$ cos\theta \in [\cos\theta_\mathrm{min}, \cos\theta_\mathrm{max}] \f$.
 * The sampling function for the angular deflection
 * \f[
 * \mu(\theta) \equiv \frac{1}{2}(1 - \cos\theta)
 * \f]
 * is
 * \f[
   \mu = \mu_1 + \frac{(A + \mu_1) \xi (\mu_2 - \mu_1)}{A + \mu_2 - \xi (\mu_2
   - \mu_1)},
 * \f]
 * where \f$ \mu_1 = \frac{1}{2}(1 - \cos\theta_\mathrm{min}) \f$,
 * \f$ \mu_2 = \frac{1}{2}(1 - \cos\theta_\mathrm{max}) \f$,
 * \f$ A \f$ is the screening coefficient, and
 * \f$ \xi \sim U(0,1) \f$.
 */
class WentzelDistribution
{
  public:
    //!@{
    //! \name Type aliases
    using MomentumSq = units::MevMomentumSq;
    using Energy = units::MevEnergy;
    using Mass = units::MevMass;
    using MottCoeffMatrix = MottElementData::MottCoeffMatrix;
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

    // Matrix of Mott coefficients for the target element
    MottCoeffMatrix const& mott_coeffs_;

    // Cosine of the minimum scattering angle
    real_type cos_thetamin_;

    // Cosine of the maximum scattering angle
    real_type cos_thetamax_;

    //// TYPES ////

    using InvMomSq = RealQuantity<UnitInverse<units::MevMomentumSq::unit_type>>;

    //// HELPER FUNCTIONS ////

    // Calculates the form factor from the scattered polar angle
    inline CELER_FUNCTION real_type calculate_form_factor(real_type cos_t) const;

    // Calculate the nuclear form momentum scale
    inline CELER_FUNCTION real_type nuclear_form_momentum_scale() const;

    // Calculate the squared momentum transfer to the target
    inline CELER_FUNCTION real_type mom_transfer_sq(real_type cos_t) const;

    // Momentum prefactor used in exponential and gaussian form factors
    inline CELER_FUNCTION InvMomSq nuclear_form_prefactor() const;

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
    , mott_coeffs_(particle_.charge() < zero_quantity()
                       ? wentzel_.mott_coeffs[el_id].electron
                       : wentzel_.mott_coeffs[el_id].positron)
    , cos_thetamin_(cos_thetamin)
    , cos_thetamax_(cos_thetamax)
{
    CELER_EXPECT(el_id < wentzel_.mott_coeffs.size());
    CELER_EXPECT(cos_thetamin_ >= -1 && cos_thetamin_ <= 1);
    CELER_EXPECT(cos_thetamax_ >= -1 && cos_thetamax_ <= 1);
    CELER_EXPECT(cos_thetamax_ <= cos_thetamin_);
}

//---------------------------------------------------------------------------//
/*!
 * Sample the cosine polar scattered angle of the incident particle.
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

        // Calculate rejection for false scattering
        MottRatioCalculator calc_mott_ratio(mott_coeffs_,
                                            std::sqrt(particle_.beta_sq()));
        real_type xs = calc_mott_ratio(cos_theta)
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
 * The models, described in \citet{leroy-2016,
 * https://doi-org.ezproxy.cern.ch/10.1142/9167} section 2.4.2.1, parameterize
 * the charge distribution inside a nucleus.
 */
CELER_FUNCTION real_type
WentzelDistribution::calculate_form_factor(real_type cos_t) const
{
    CELER_EXPECT(cos_t >= -1 && cos_t <= 1);

    using MomSq = NuclearFormFactorTraits::MomentumSq;

    if (wentzel_.params.form_factor_type == NuclearFormFactorType::none)
    {
        return 1;
    }

    // Approximate squared momentum transfer to the target (assuming heavy
    // target and light projectile)
    auto mt_sq
        = MomSq{2 * value_as<MomSq>(particle_.momentum_sq()) * (1 - cos_t)};

    auto calc_ff = [mt_sq](auto&& calc_form_factor) {
        real_type result = calc_form_factor(mt_sq);
        CELER_ENSURE(result >= 0 && result <= 1);
        return result;
    };

    switch (wentzel_.params.form_factor_type)
    {
        case NuclearFormFactorType::flat:
            return calc_ff(UUNuclearFormFactor{target_.atomic_mass_number()});
        case NuclearFormFactorType::exponential:
            return calc_ff(
                ExpNuclearFormFactor{this->nuclear_form_prefactor()});
        case NuclearFormFactorType::gaussian:
            return calc_ff(
                GaussianNuclearFormFactor{this->nuclear_form_prefactor()});
        default:
            CELER_ASSERT_UNREACHABLE();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the flat form factor.
 *
 * See \cite{leroy-2016} Eq. 2.265.
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
 * Eq. 2.265 of \cite{leroy-2016}.
 */
CELER_CONSTEXPR_FUNCTION real_type WentzelDistribution::flat_coeff()
{
    using namespace celeritas::units::literals;
    return native_value_to<units::MevMomentum>(2.0_fm / constants::hbar_planck)
        .value();
}

//---------------------------------------------------------------------------//
/*!
 * Get the constant prefactor of the squared momentum transfer.
 */
CELER_FUNCTION auto WentzelDistribution::nuclear_form_prefactor() const
    -> InvMomSq
{
    CELER_EXPECT(target_.isotope_id() < wentzel_.nuclear_form_prefactor.size());
    return InvMomSq{wentzel_.nuclear_form_prefactor[target_.isotope_id()]};
}

//---------------------------------------------------------------------------//
/*!
 * Sample the scattering polar angle of the incident particle.
 */
template<class Engine>
CELER_FUNCTION real_type WentzelDistribution::sample_costheta(
    real_type cos_thetamin, real_type cos_thetamax, Engine& rng) const
{
    // Sample scattering angle [fern] eqn 92, where cos(theta) = 1 - 2*mu
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
