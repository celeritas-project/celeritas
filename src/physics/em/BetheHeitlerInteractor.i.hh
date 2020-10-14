//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file BetheHeitlerInteractor.i.hh
//---------------------------------------------------------------------------//

#include "base/ArrayUtils.hh"
#include "base/Constants.hh"
#include "random/distributions/BernoulliDistribution.hh"
#include "random/distributions/GenerateCanonical.hh"
#include "random/distributions/UniformRealDistribution.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 *
 * The incident particle must be above the energy threshold: this should be
 * handled in code *before* the interactor is constructed.
 */
BetheHeitlerInteractor::BetheHeitlerInteractor(
    const BetheHeitlerInteractorPointers& shared,
    const ParticleTrackView&              particle,
    const Real3&                          inc_direction,
    SecondaryAllocatorView&               allocate,
    const MaterialMock&                   material)
    : shared_(shared)
    , inc_energy_(particle.energy().value())
    , inc_direction_(inc_direction)
    , allocate_(allocate)
    , material_(material)
{
    REQUIRE(particle.def_id() == shared_.gamma_id);
    REQUIRE(inc_energy_ >= this->min_incident_energy()
            && inc_energy_ <= this->max_incident_energy());

    epsilon0_ = 1 / (shared_.inv_electron_mass * inc_energy_.value());
}

//---------------------------------------------------------------------------//
/*!
 * Pair-production using the Bethe-Heitler model.
 *
 * See section 6.5 of the Geant physics reference 10.6.
 */
template<class Engine>
CELER_FUNCTION Interaction BetheHeitlerInteractor::operator()(Engine& rng)
{
    // Allocate space for the pair-produced electrons
    Secondary* electron_pair = this->allocate_(2);
    if (electron_pair == nullptr)
    {
        // Failed to allocate space for a secondary
        return Interaction::from_failure();
    }

    // Minimum (\epsilon = 0.5) and maximum (\epsilon = \epsilon_1) values of
    // screening variable, \delta.
    real_type delta_min = 136.0 * std::pow(material_.Z(), -1 / 3) * 4.0
                          * epsilon0_;
    real_type delta_max = std::exp((42.24 - material_.Z()) / 8.368) - 0.952;

    // Determine kinematical limits on \epsilon.
    real_type epsilon_1   = 0.5 - 0.5 * std::sqrt(1 - delta_min / delta_max);
    real_type epsilon_min = std::max(epsilon0_, epsilon_1);

    // Decide to choose f1, g1 or f2, g2 based on N1, N2 (factors from
    // corrected Bethe-Heitler cross section; c.f. Eq. 6.6 of Geant4 Physics
    // Reference 10.6)
    BernoulliDistribution choose_f1g1(
        (epsilon_min * epsilon_min - epsilon_min
         + 0.25 * this->Phi1_aux(delta_min, material_.Z()))
        / (1.5 * this->Phi2_aux(delta_min, material_.Z())));

    // Sample epsilon
    real_type epsilon;
    // Temporary sample values used in rejection
    real_type reject_threshold;
    do
    {
        if (choose_f1g1(rng))
        {
            // Used to sample from f1
            epsilon = 0.5
                      - (0.5 - epsilon_min)
                            * std::pow(generate_canonical(rng), 1 / 3);
            real_type epsilon_sq = epsilon * epsilon;
            // Calculate f1 density function
            real_type f1 = 3 / std::pow(0.5 - epsilon_min, 3)
                           * (epsilon_sq - epsilon + 0.25);
            // Calculate delta from material Z and sampled epsilon
            real_type d = this->delta(material_.Z(), epsilon);
            // Calculate g1 "rejection" function
            reject_threshold = this->Phi1_aux(d, material_.Z())
                               / this->Phi1_aux(delta_min, material_.Z());
        }
        else
        {
            // Used to sample from f2
            epsilon = epsilon_min + (0.5 - epsilon_min) * rnd_b(rng);
            // Calculate f2 density function (constant)
            real_type f2 = 1 / (0.5 - epsilon_min);
            // Calculate delta given the material Z and sampled epsilon
            real_type d = this->delta(material_.Z(), epsilon);
            // Calculate g2 "rejection" function
            reject_threshold = this->Phi2_aux(d, material_.Z())
                               / this->Phi2_aux(delta_min, material_.Z());
        }
        CHECK(epsilon >= epsilon_min && epsilon <= 0.5);
    } while (generate_canonical(rng) > reject_threshold);

    // Construct interaction for change to primary (incident) particle
    Interaction result;
    result.action      = Action::absorbed;
    result.energy      = units::MevEnergy{0};
    result.direction   = {0, 0, 0};
    result.secondaries = {electron_pair, 2};

    // Argument to density function for polar angle
    real_type density_arg;
    real_type log_r2r3
        = std::log(generate_canonical(rng) * generate_canonical(rng));
    if (generate_canonical(rng) < 0.25)
    {
        density_arg = log_r2r3 * 1.6;
    }
    else
    {
        density_arg = log_r2r3 / 1.875;
    }

    // Polar angle from the density function
    real_type theta = 0.0977
                      * (density_arg * std::exp(-log_r2r3)
                         + 27 * density_arg * std::exp(-log_r2r3));

    // Azimuthal angle sampled isotropically
    UniformRealDistribution<real_type> sample_phi(0, 2 * constants::pi);
    electron_pair[0].direction = rotate(
        from_spherical(std::cos(theta), sample_phi(rng)), inc_direction_);
    electron_pair[1].direction = rotate(
        from_spherical(-std::cos(theta), sample_phi(rng)), inc_direction_);

    // Outgoing secondaries are electron and positron
    electron_pair[0].def_id = shared_.electron_id;
    electron_pair[1].def_id = shared_.positron_id;
    // Energies of secondaries shared equally
    electron_pair[0].energy
        = units::MevEnergy{0.5 * epsilon * inc_energy_.value()};
    electron_pair[1].def_id
        = units::MevEnergy{0.5 * epsilon * inc_energy_.value()};
    // Calculate produced pair directions via conservation of momentum
    for (unsigned int p = 0; p < allocate_.capacity(); p++)
    {
        for (int i = 0; i < 3; ++i)
        {
            electron_pair[p].direction[i]
                = inc_direction_[i] * inc_energy_.value()
                  - electron_pair[p].direction[i]
                        * electron_pair[p].energy.value();
        }
        normalize_direction(&electron_pair[p].direction);
    }

    return result;
}

CELER_FUNCTION real_type BetheHeitlerInteractor::delta(size_type Z,
                                                       real_type eps) const
{
    return 136 * std::pow(Z, -1 / 3) * epsilon0_ * epsilon0_
           / (eps * (1 - eps));
}

CELER_FUNCTION real_type BetheHeitlerInteractor::Phi1(real_type delta) const
{
    if (delta <= 1)
    {
        return 20.867 - 3.242 * delta + 0.625 * delta * delta;
    }
    return Phi12(delta);
}

CELER_FUNCTION real_type BetheHeitlerInteractor::Phi2(real_type delta) const
{
    if (delta <= 1)
    {
        return 20.209 - 1.930 * delta - 0.086 * delta * delta;
    }
    return Phi12(delta);
}

CELER_FUNCTION real_type BetheHeitlerInteractor::Phi12(real_type delta) const
{
    return 21.12 - 4.184 * std::log(delta + 0.952);
}

CELER_FUNCTION real_type BetheHeitlerInteractor::CoulombCorr(size_type Z) const
{
    real_type term1 = 8 / 3 * std::log(Z);

    if (inc_energy_.value() < 50.0) // 50 MeV
    {
        return term1;
    }

    return term1 + 8 * this->CoulombCorr_aux(Z);
}

CELER_FUNCTION real_type BetheHeitlerInteractor::CoulombCorr_aux(size_type Z) const
{
    real_type alphaZ_sq = constants::alpha_fine_structure
                          * constants::alpha_fine_structure * Z * Z;
    real_type alphaZ_sqsq = alphaZ_sq * alphaZ_sq;
    return alphaZ_sq
           * (1 / (1 + alphaZ_sq) + 0.20206 - 0.0369 * alphaZ_sq
              + 0.0083 * alphaZ_sqsq - 0.0020 * alphaZ_sq * alphaZ_sqsq);
}

CELER_FUNCTION real_type BetheHeitlerInteractor::Phi1_aux(real_type delta,
                                                          size_type Z) const
{
    return (3 * this->Phi1(delta) - this->Phi2(delta) - this->CoulombCorr(Z));
}

CELER_FUNCTION real_type BetheHeitlerInteractor::Phi2_aux(real_type delta,
                                                          size_type Z) const
{
    return (3 / 2 * this->Phi1(delta) - 0.5 * this->Phi2(delta)
            - this->CoulombCorr(Z));
}

// CELER_FUNCTION real_type BetheHeitler::polar_angle_density() const
// {
//     return 0 .0;
// }

//---------------------------------------------------------------------------//
} // namespace celeritas
