//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file EnergyLossDistribution.i.hh
//---------------------------------------------------------------------------//

#include "base/Assert.hh"
#include "random/distributions/GammaDistribution.hh"
#include "random/distributions/PoissonDistribution.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from model parameters, incident particle, and mean energy loss.
 */
CELER_FUNCTION
EnergyLossDistribution::EnergyLossDistribution(const FluctuationData& shared,
                                               const CutoffView&      cutoffs,
                                               const MaterialTrackView& material,
                                               const ParticleTrackView& particle,
                                               MevEnergy mean_loss,
                                               real_type step_length)
    : shared_(shared)
    , material_(material)
    , mean_loss_(mean_loss.value())
    , step_length_(step_length)
    , mass_ratio_(shared_.electron_mass / particle.mass().value())
    , charge_sq_(ipow<2>(particle.charge().value()))
    , gamma_(particle.lorentz_factor())
    , gamma_sq_(ipow<2>(gamma_))
    , beta_sq_(1 - (1 / gamma_sq_))
    , max_energy_transfer_(
          particle.particle_id() == shared_.electron_id
              ? particle.energy().value() * real_type(0.5)
              : 2 * shared_.electron_mass * beta_sq_ * gamma_sq_
                    / (1 + mass_ratio_ * (2 * gamma_ + mass_ratio_)))
    , max_energy_(min(cutoffs.energy(shared_.electron_id).value(),
                      max_energy_transfer_))
{
    CELER_EXPECT(mean_loss_ > 0);
    CELER_EXPECT(step_length_ > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Sample the actual energy loss from the mean loss.
 */
template<class Engine>
CELER_FUNCTION auto EnergyLossDistribution::operator()(Engine& rng) const
    -> MevEnergy
{
    // Small step or low density material
    if (mean_loss_ < EnergyLossDistribution::min_valid_energy().value()
        || max_energy_ <= EnergyLossDistribution::ionization_energy().value())
    {
        return MevEnergy{mean_loss_};
    }

    // The Gaussian approximation is valid for heavy particles and in the
    // regime \f$ \kappa > 10 \f$. Fluctuations of the unrestricted energy loss
    // follow a Gaussian distribution if \f$ \Delta E > \kappa T_{max} \f$,
    // where \f$ T_{max} \f$ is the maximum energy transfer (PHYS332 section
    // 2). For fluctuations of the \em restricted energy loss, the condition is
    // modified to \f$ \Delta E > \kappa T_{c} \f$ and \f$ T_{max} \le 2 T_c
    // \f$, where \f$ T_c \f$ is the delta ray cutoff energy (PRM Eq. 7.6-7.7).
    if (mass_ratio_ < 1
        && mean_loss_ >= EnergyLossDistribution::min_kappa() * max_energy_
        && max_energy_transfer_ <= 2 * max_energy_)
    {
        // Approximate straggling function as a Gaussian distribution
        return MevEnergy{this->sample_gaussian(rng)};
    }
    // Use Urban model of energy loss fluctuations in thin layers
    return MevEnergy{this->sample_urban(rng)};
}

//---------------------------------------------------------------------------//
/*!
 * Gaussian model of energy loss fluctuations.
 *
 * In a thick absorber, the total energy transfer is a result of many small
 * energy losses from a large number of collisions. The central limit theorem
 * applies, and the energy loss fluctuations can be described by a Gaussian
 * distribution. See section 7.3.1 of the Geant4 Physics Reference Manual and
 * GEANT3 PHYS332 section 2.3.
 */
template<class Engine>
CELER_FUNCTION real_type EnergyLossDistribution::sample_gaussian(Engine& rng) const
{
    // Square root of Bohr's variance (PRM Eq. 7.8). For thick absorbers, the
    // straggling function approaches a Gaussian distribution with this
    // standard deviation
    const real_type stddev = std::sqrt(
        2 * constants::pi * ipow<2>(constants::r_electron)
        * shared_.electron_mass * material_.material_view().electron_density()
        * charge_sq_ * max_energy_ * step_length_
        * (1 / beta_sq_ - real_type(0.5)));

    // Sample energy loss from a Gaussian distribution
    if (mean_loss_ >= 2 * stddev)
    {
        real_type                     result;
        const real_type               max_loss = 2 * mean_loss_;
        NormalDistribution<real_type> sample_normal(mean_loss_, stddev);
        do
        {
            result = sample_normal(rng);
        } while (result <= 0 || result > max_loss);
        return result;
    }
    // Sample energy loss from a gamma distribution. Note that while this
    // appears in G4UniversalFluctuation, the Geant4 documentation does not
    // explain why the loss is sampled from a gamma distribution in this case.
    const real_type k = ipow<2>(mean_loss_ / stddev);
    return GammaDistribution<real_type>(k, mean_loss_ / k)(rng);
}

//---------------------------------------------------------------------------//
/*!
 * Urban model of energy loss fluctuations in thin layers.
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
template<class Engine>
CELER_FUNCTION real_type EnergyLossDistribution::sample_urban(Engine& rng) const
{
    // Material-dependent data
    const auto  mat            = material_.material_view();
    const auto& params         = shared_.params[material_.material_id()];
    auto        binding_energy = params.binding_energy;

    // Width correction: the FWHM of the energy loss distribution in thin
    // layers is in most cases too small; this width correction is used to get
    // more accurate FWHM values while keeping the mean loss the same by
    // rescaling the energy levels and number of excitations. The width
    // correction algorithm is discussed (though not in much detail) in PRM
    // section 7.3.3

    const real_type loss_scaling = min(1 + 5e-4 / max_energy_, real_type(1.5));
    const real_type mean_loss    = mean_loss_ / loss_scaling;

    // Calculate the excitation macroscopic cross sections and apply the width
    // correction
    Real2 xs_exc{0, 0};
    if (max_energy_ > mat.mean_excitation_energy().value())
    {
        // Common term in the numerator and denominator of PRM Eq. 7.10
        const real_type w
            = std::log(2 * shared_.electron_mass * beta_sq_ * gamma_sq_)
              - beta_sq_;
        if (w > mat.log_mean_excitation_energy().value())
        {
            if (w > params.log_binding_energy[1])
            {
                const real_type c
                    = mean_loss * (1 - EnergyLossDistribution::rate())
                      / (w - mat.log_mean_excitation_energy().value());
                for (int i : range(2))
                {
                    // Excitation macroscopic cross section (PRM Eq. 7.10)
                    xs_exc[i] = c * params.oscillator_strength[i]
                                * (w - params.log_binding_energy[i])
                                / params.binding_energy[i];
                }
            }
            else
            {
                xs_exc[0] = mean_loss * (1 - EnergyLossDistribution::rate())
                            / params.binding_energy[0];
            }

            // Scale the binding energy and macroscopic cross section (i.e.,
            // the mean number of excitations)
            real_type scaling = 4;
            if (xs_exc[0] < EnergyLossDistribution::exc_thresh())
            {
                scaling = real_type(0.5)
                          + (scaling - real_type(0.5))
                                * std::sqrt(
                                    xs_exc[0]
                                    / EnergyLossDistribution::exc_thresh());
            }
            binding_energy[0] *= scaling;
            xs_exc[0] /= scaling;
        }
    }

    // Calculate the ionization macroscopic cross section (PRM Eq. 7.11)
    constexpr real_type e_0
        = EnergyLossDistribution::ionization_energy().value();
    real_type xs_ion = mean_loss * (max_energy_ - e_0)
                       / (max_energy_ * e_0 * std::log(max_energy_ / e_0));
    if (xs_exc[0] + xs_exc[1] > 0)
    {
        // The contribution from excitation is nonzero, so scale the ionization
        // cross section accordingly
        xs_ion *= EnergyLossDistribution::rate();
    }

    // Calculate actual energy loss from the loss contributions from excitation
    // and ionization
    const real_type result
        = loss_scaling
          * (this->sample_excitation_loss(xs_exc, binding_energy, rng)
             + this->sample_ionization_loss(xs_ion, rng));

    CELER_ENSURE(result > 0);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the energy loss contribution from excitation for the Urban model.
 */
template<class Engine>
CELER_FUNCTION real_type EnergyLossDistribution::sample_excitation_loss(
    Real2 xs, Real2 binding_energy, Engine& rng) const
{
    real_type result = 0;

    // Mean and variance for fast sampling from Gaussian
    real_type mean     = 0;
    real_type variance = 0;

    for (int i : range(2))
    {
        if (xs[i] > EnergyLossDistribution::max_collisions())
        {
            // WHen the number of collisions is large, use faster approach
            // of sampling from a Gaussian
            mean += xs[i] * binding_energy[i];
            variance += xs[i] * ipow<2>(binding_energy[i]);
        }
        else if (xs[i] > 0)
        {
            // The loss due to excitation is \f$ \Delta E_{exc} = n_1 E_1 + n_2
            // E_2 \f$, where the number of collisions \f$ n_i \f$ is sampled
            // from a Poisson distribution with mean \f$ \Sigma_i \f$
            auto n = PoissonDistribution<real_type>(xs[i])(rng);
            if (n > 0)
            {
                result += ((n + 1) - 2 * generate_canonical(rng))
                          * binding_energy[i];
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
EnergyLossDistribution::sample_ionization_loss(real_type xs, Engine& rng) const
{
    real_type result = 0;

    constexpr real_type e_0
        = EnergyLossDistribution::ionization_energy().value();
    const real_type energy_ratio = max_energy_ / e_0;

    // Parameter that determines the upper limit of the energy interval in
    // which the fast simulation is used
    real_type alpha = 1;

    // Mean number of collisions in the fast simulation interval
    real_type mean_num_coll = 0;

    if (xs > EnergyLossDistribution::max_collisions())
    {
        // When the number of collisions is large, fast sampling from a
        // Gaussian is used in the lower portion of the energy loss interval.
        // See PHYS332 section 2.4: Fast simulation for \f$ n_3 \ge 16 \f$

        // Calculate the maximum value of \f$ \alpha \f$ (Eq. 25)
        alpha
            = (xs + EnergyLossDistribution::max_collisions()) * energy_ratio
              / (EnergyLossDistribution::max_collisions() * energy_ratio + xs);

        // Mean energy loss for a single collision of this type (Eq. 14)
        const real_type mean_loss_coll = alpha * std::log(alpha) / (alpha - 1);

        // Mean number of collisions of this type (Eq. 16)
        mean_num_coll = xs * energy_ratio * (alpha - 1)
                        / ((energy_ratio - 1) * alpha);

        // Mean and standard deviation of the total energy loss (Eqs. 18-19)
        const real_type mean = mean_num_coll * mean_loss_coll * e_0;
        const real_type stddev
            = e_0 * std::sqrt(xs * (alpha - ipow<2>(mean_loss_coll)));

        // Sample energy loss from a Gaussian distribution
        result += this->sample_fast_urban(mean, stddev, rng);
    }
    if (xs > 0 && energy_ratio > alpha)
    {
        // Sample number of ionizations from a Poisson distribution with mean
        // \f$ n_3 - n_A \f$, where \f$ n_3 \f$ is the number of ionizations
        // and \f$ n_A \f$ is the number of ionizations in the energy interval
        // in which the fast sampling from a Gaussian is used
        auto n = PoissonDistribution<real_type>(xs - mean_num_coll)(rng);

        // Add the contribution from ionizations in the energy interval in
        // which the energy loss is sampled for each collision (Eq. 20)
        const real_type w = (max_energy_ - alpha * e_0) / max_energy_;
        for (CELER_MAYBE_UNUSED int i : range(n))
        {
            result += alpha * e_0 / (1 - w * generate_canonical(rng));
        }
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Fast sampling of energy loss in thin absorbers from a Gaussian distribution.
 */
template<class Engine>
CELER_FUNCTION real_type EnergyLossDistribution::sample_fast_urban(
    real_type mean, real_type stddev, Engine& rng) const
{
    if (stddev <= 4 * mean)
    {
        const real_type               max_loss = 2 * mean;
        NormalDistribution<real_type> sample_normal(mean, stddev);
        do
        {
            mean = sample_normal(rng);
        } while (mean <= 0 || mean > max_loss);
        return mean;
    }
    return 2 * mean * generate_canonical(rng);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
