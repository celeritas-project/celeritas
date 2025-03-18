//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/distribution/EnergyLossUrbanDistribution.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/em/data/FluctuationData.hh"
#include "celeritas/random/distribution/PoissonDistribution.hh"
#include "celeritas/random/distribution/UniformRealDistribution.hh"

#include "EnergyLossGaussianDistribution.hh"
#include "EnergyLossHelper.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Sample from the Urban model of energy loss fluctuations in thin layers.
 *
 * The Urban model is used to compute the energy loss fluctuation when \f$
 * \kappa \f$ is small. It assumes atoms have only two energy levels with
 * binding energies \f$ E_1 \f$ and \f$ E_2 \f$ and that the particle-atom
 * interaction will either be an excitation with energy loss \f$ E_1 \f$ or \f$
 * E_2 \f$ or an ionization with energy loss proportional to \f$ 1 / E^2 \f$.
 * The number of collisions for each interaction type has a Poisson
 * distribution with mean proportional to the interaction cross section. The
 * energy loss in a step will be the sum of the energy loss contributions from
 * excitation and ionization.
 *
 * When the number of ionizations is larger than a given threshold, a fast
 * sampling method can be used instead. The possible energy loss interval is
 * divided into two parts: a lower range in which the number of collisions is
 * large and the energy loss can be sampled from a Gaussian distribution, and
 * an upper range in which the energy loss is sampled for each collision.
 *
 * See section 7.3.2 of the Geant4 Physics Reference Manual and GEANT3 PHYS332
 * section 2.4 for details.
 */
class EnergyLossUrbanDistribution
{
  public:
    //!@{
    //! \name Type aliases
    using FluctuationRef = NativeCRef<FluctuationData>;
    using Energy = units::MevEnergy;
    using Mass = units::MevMass;
    //!@}

  public:
    // Construct from particle properties
    inline CELER_FUNCTION
    EnergyLossUrbanDistribution(FluctuationRef const& shared,
                                MaterialTrackView const& cur_mat,
                                Energy unscaled_mean_loss,
                                Energy max_energy,
                                Mass two_mebsgs,
                                real_type beta_sq);

    // Construct from helper-calculated data
    explicit inline CELER_FUNCTION
    EnergyLossUrbanDistribution(EnergyLossHelper const& helper);

    // Sample energy loss according to the distribution
    template<class Generator>
    inline CELER_FUNCTION Energy operator()(Generator& rng);

  private:
    //// TYPES ////

    using Real2 = Array<real_type, 2>;

    //// DATA ////

    real_type max_energy_;
    real_type loss_scaling_;
    Real2 binding_energy_;
    Real2 xs_exc_;
    real_type xs_ion_;

    //// CONSTANTS ////

    //! Relative contribution of ionization to energy loss
    static CELER_CONSTEXPR_FUNCTION real_type rate() { return 0.56; }

    //! Number of collisions above which to use faster sampling from Gaussian
    static CELER_CONSTEXPR_FUNCTION size_type max_collisions() { return 8; }

    //! Threshold number of excitations used in width correction
    static CELER_CONSTEXPR_FUNCTION real_type exc_thresh() { return 42; }

    //! Ionization energy [MeV]
    static CELER_CONSTEXPR_FUNCTION real_type ionization_energy()
    {
        return value_as<Energy>(EnergyLossHelper::ionization_energy());
    }

    //! Energy point below which FWHM scaling doesn't change [MeV]
    static CELER_CONSTEXPR_FUNCTION real_type fwhm_min_energy()
    {
        return 1e-3;  // 1 keV
    }

    //// HELPER FUNCTIONS ////

    template<class Engine>
    CELER_FUNCTION real_type sample_excitation_loss(Engine& rng);

    template<class Engine>
    CELER_FUNCTION real_type sample_ionization_loss(Engine& rng);

    template<class Engine>
    static CELER_FUNCTION real_type sample_fast_urban(real_type mean,
                                                      real_type stddev,
                                                      Engine& rng);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from distribution parameters.
 */
CELER_FUNCTION EnergyLossUrbanDistribution::EnergyLossUrbanDistribution(
    FluctuationRef const& shared,
    MaterialTrackView const& cur_mat,
    Energy unscaled_mean_loss,
    Energy max_energy,
    Mass two_mebsgs,
    real_type beta_sq)
    : max_energy_(max_energy.value())
{
    CELER_EXPECT(unscaled_mean_loss > zero_quantity());
    CELER_EXPECT(two_mebsgs > zero_quantity());
    CELER_EXPECT(beta_sq > 0);

    // Width correction: the FWHM of the energy loss distribution in thin
    // layers is in most cases too small; this width correction is used to get
    // more accurate FWHM values while keeping the mean loss the same by
    // rescaling the energy levels and number of excitations. The width
    // correction algorithm is discussed (though not in much detail) in PRM
    // section 7.3.3
    loss_scaling_
        = real_type(0.5)
              * min(this->fwhm_min_energy() / max_energy_, real_type(1))
          + real_type(1);
    real_type const mean_loss = unscaled_mean_loss.value() / loss_scaling_;

    // Material-dependent data
    CELER_ASSERT(cur_mat.material_id() < shared.urban.size());
    UrbanFluctuationParameters const& params
        = shared.urban[cur_mat.material_id()];
    binding_energy_ = params.binding_energy;
    xs_exc_ = {0, 0};

    // Calculate the excitation macroscopic cross sections and apply the width
    // correction
    auto const& mat = cur_mat.material_record();
    if (max_energy_ > value_as<Energy>(mat.mean_excitation_energy()))
    {
        // Common term in the numerator and denominator of PRM Eq. 7.10
        // two_mebsgs = 2 * m_e c^2 * beta^2 * gamma^2
        real_type const w = std::log(value_as<units::MevMass>(two_mebsgs))
                            - beta_sq;
        real_type const w_0
            = value_as<units::LogMevEnergy>(mat.log_mean_excitation_energy());
        if (w > w_0)
        {
            if (w > params.log_binding_energy[1])
            {
                real_type const c = mean_loss * (1 - this->rate()) / (w - w_0);
                for (int i : range(2))
                {
                    // Excitation macroscopic cross section (PRM Eq. 7.10)
                    xs_exc_[i] = c * params.oscillator_strength[i]
                                 * (w - params.log_binding_energy[i])
                                 / params.binding_energy[i];
                }
            }
            else
            {
                xs_exc_[0] = mean_loss * (1 - this->rate())
                             / params.binding_energy[0];
            }

            // Scale the binding energy and macroscopic cross section (i.e.,
            // the mean number of excitations)
            real_type scaling = 4;
            if (xs_exc_[0] < this->exc_thresh())
            {
                scaling = real_type(0.5)
                          + (scaling - real_type(0.5))
                                * std::sqrt(xs_exc_[0] / this->exc_thresh());
            }
            binding_energy_[0] *= scaling;
            xs_exc_[0] /= scaling;
        }
    }

    // Calculate the ionization macroscopic cross section (PRM Eq. 7.11)
    constexpr real_type e_0 = EnergyLossUrbanDistribution::ionization_energy();
    xs_ion_ = mean_loss * (max_energy_ - e_0)
              / (max_energy_ * e_0 * std::log(max_energy_ / e_0));
    if (xs_exc_[0] + xs_exc_[1] > 0)
    {
        // The contribution from excitation is nonzero, so scale the ionization
        // cross section accordingly
        xs_ion_ *= this->rate();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Construct from helper-calculated data.
 */
CELER_FUNCTION EnergyLossUrbanDistribution::EnergyLossUrbanDistribution(
    EnergyLossHelper const& helper)
    : EnergyLossUrbanDistribution(helper.shared(),
                                  helper.material(),
                                  helper.mean_loss(),
                                  helper.max_energy(),
                                  helper.two_mebsgs(),
                                  helper.beta_sq())
{
}

//---------------------------------------------------------------------------//
/*!
 * Sample energy loss according to the distribution.
 */
template<class Generator>
CELER_FUNCTION auto
EnergyLossUrbanDistribution::operator()(Generator& rng) -> Energy
{
    // Calculate actual energy loss from the loss contributions from excitation
    // and ionization
    real_type result = this->sample_excitation_loss(rng)
                       + this->sample_ionization_loss(rng);

    CELER_ENSURE(result >= 0);
    return Energy{loss_scaling_ * result};
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the energy loss contribution from excitation for the Urban model.
 */
template<class Engine>
CELER_FUNCTION real_type
EnergyLossUrbanDistribution::sample_excitation_loss(Engine& rng)
{
    real_type result = 0;

    // Mean and variance for fast sampling from Gaussian
    real_type mean = 0;
    real_type variance = 0;

    for (int i : range(2))
    {
        if (xs_exc_[i] > this->max_collisions())
        {
            // When the number of collisions is large, use faster approach
            // of sampling from a Gaussian
            mean += xs_exc_[i] * binding_energy_[i];
            variance += xs_exc_[i] * ipow<2>(binding_energy_[i]);
        }
        else if (xs_exc_[i] > 0)
        {
            // The loss due to excitation is \f$ \Delta E_{exc} = n_1 E_1 + n_2
            // E_2 \f$, where the number of collisions \f$ n_i \f$ is sampled
            // from a Poisson distribution with mean \f$ \Sigma_i \f$
            unsigned int n = PoissonDistribution<real_type>(xs_exc_[i])(rng);
            if (n > 0)
            {
                UniformRealDistribution<real_type> sample_fraction(n - 1,
                                                                   n + 1);
                result += sample_fraction(rng) * binding_energy_[i];
            }
        }
    }
    if (variance > 0)
    {
        // Sample excitation energy loss contribution from a Gaussian
        result += this->sample_fast_urban(mean, std::sqrt(variance), rng);
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the energy loss contribution from ionization for the Urban model.
 */
template<class Engine>
CELER_FUNCTION real_type
EnergyLossUrbanDistribution::sample_ionization_loss(Engine& rng)
{
    real_type result = 0;

    constexpr real_type e_0 = EnergyLossUrbanDistribution::ionization_energy();
    real_type const energy_ratio = max_energy_ / e_0;

    // Parameter that determines the upper limit of the energy interval in
    // which the fast simulation is used
    real_type alpha = 1;

    // Mean number of collisions in the fast simulation interval
    real_type mean_num_coll = 0;

    if (xs_ion_ > this->max_collisions())
    {
        // When the number of collisions is large, fast sampling from a
        // Gaussian is used in the lower portion of the energy loss interval.
        // See PHYS332 section 2.4: Fast simulation for \f$ n_3 \ge 16 \f$

        // Calculate the maximum value of \f$ \alpha \f$ (Eq. 25)
        alpha = (xs_ion_ + this->max_collisions()) * energy_ratio
                / (this->max_collisions() * energy_ratio + xs_ion_);

        // Mean energy loss for a single collision of this type (Eq. 14)
        real_type const mean_loss_coll = alpha * std::log(alpha) / (alpha - 1);

        // Mean number of collisions of this type (Eq. 16)
        mean_num_coll = xs_ion_ * energy_ratio * (alpha - 1)
                        / ((energy_ratio - 1) * alpha);

        // Mean and standard deviation of the total energy loss (Eqs. 18-19)
        real_type const mean = mean_num_coll * mean_loss_coll * e_0;
        real_type const stddev
            = e_0 * std::sqrt(xs_ion_ * (alpha - ipow<2>(mean_loss_coll)));

        // Sample energy loss from a Gaussian distribution
        result += this->sample_fast_urban(mean, stddev, rng);
    }
    if (xs_ion_ > 0 && energy_ratio > alpha)
    {
        // Sample number of ionizations from a Poisson distribution with mean
        // \f$ n_3 - n_A \f$, where \f$ n_3 \f$ is the number of ionizations
        // and \f$ n_A \f$ is the number of ionizations in the energy interval
        // in which the fast sampling from a Gaussian is used
        PoissonDistribution<real_type> sample_num_ioni(xs_ion_ - mean_num_coll);

        // Add the contribution from ionizations in the energy interval in
        // which the energy loss is sampled for each collision (Eq. 20)
        UniformRealDistribution<real_type> sample_fraction(
            alpha / energy_ratio, 1);
        for (auto n = sample_num_ioni(rng); n > 0; --n)
        {
            result += alpha * e_0 / sample_fraction(rng);
        }
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Fast sampling of energy loss in thin absorbers from a Gaussian distribution.
 */
template<class Engine>
CELER_FUNCTION real_type EnergyLossUrbanDistribution::sample_fast_urban(
    real_type mean, real_type stddev, Engine& rng)
{
    if (stddev <= 4 * mean)
    {
        EnergyLossGaussianDistribution sample_eloss(Energy{mean},
                                                    Energy{stddev});
        return value_as<Energy>(sample_eloss(rng));
    }
    else
    {
        UniformRealDistribution<real_type> sample(0, 2 * mean);
        return sample(rng);
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
