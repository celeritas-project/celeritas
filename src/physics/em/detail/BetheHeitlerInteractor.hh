//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file BetheHeitlerInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Algorithms.hh"
#include "base/ArrayUtils.hh"
#include "base/Constants.hh"
#include "base/Macros.hh"
#include "base/StackAllocator.hh"
#include "base/Types.hh"
#include "physics/base/Interaction.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/Secondary.hh"
#include "physics/base/Units.hh"
#include "physics/material/ElementView.hh"
#include "random/distributions/BernoulliDistribution.hh"
#include "random/distributions/GenerateCanonical.hh"
#include "random/distributions/UniformRealDistribution.hh"

#include "BetheHeitlerData.hh"
#include "TsaiUrbanDistribution.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Bethe-Heitler model for gamma -> e+e- (electron-pair production).
 *
 * Give an incident gamma, it adds a two pair-produced secondary electrons to
 * the secondary stack. No cutoffs are performed on the incident gamma energy.
 *
 * \note This performs the same sampling routine as in Geant4's
 * G4BetheHeitlerModel, as documented in section 6.5 of the Geant4 Physics
 * Reference (release 10.6), applicable to incident gammas with energy
 * \f$ E_gamma \leq 100\f$ GeV . For \f$ E_gamma > 80 \f$ GeV, it is suggested
 * to use `G4PairProductionRelModel`.
 */
class BetheHeitlerInteractor
{
  public:
    //!@{
    //! Type aliases
    using MevMass = units::MevMass;
    //!@}

  public:
    //! Construct sampler from shared and state data
    inline CELER_FUNCTION
    BetheHeitlerInteractor(const BetheHeitlerData&    shared,
                           const ParticleTrackView&   particle,
                           const Real3&               inc_direction,
                           StackAllocator<Secondary>& allocate,
                           const ElementView&         element);

    // Sample an interaction with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng);

  private:
    //// DATA ////

    // Gamma energy divided by electron mass * csquared
    const BetheHeitlerData& shared_;
    // Incident gamma energy
    const units::MevEnergy inc_energy_;
    // Incident direction
    const Real3& inc_direction_;
    // Allocate space for a secondary particle
    StackAllocator<Secondary>& allocate_;
    // Element properties for calculating screening functions and variables
    const ElementView& element_;
    // Cached minimum epsilon, m_e*c^2/E_gamma; kinematical limit for Y -> e+e-
    real_type epsilon0_;

    //// HELPER FUNCTIONS ////

    // Calculates the screening variable, \$f \delta \eta \$f, which is a
    // function of \$f \epsilon \$f. This is a measure of the "impact
    // parameter" of the incident photon.
    inline CELER_FUNCTION real_type impact_parameter(real_type eps) const;

    // Screening function, Phi_1, for the corrected Bethe-Heitler
    // cross-section calculation.
    inline CELER_FUNCTION real_type
    screening_phi1(real_type impact_parameter) const;

    // Screening function, Phi_2, for the corrected Bethe-Heitler
    // cross-section calculation.
    inline CELER_FUNCTION real_type
    screening_phi2(real_type impact_parameter) const;

    // Auxiliary screening function, Phi_1, for the "composition+rejection"
    // technique for sampling.
    inline CELER_FUNCTION real_type screening_phi1_aux(real_type delta) const;

    // Auxiliary screening function, Phi_2, for the "composition+rejection"
    // technique for sampling.
    inline CELER_FUNCTION real_type screening_phi2_aux(real_type delta) const;
};

//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 *
 * The incident gamma energy must be at least twice the electron rest mass.
 */
BetheHeitlerInteractor::BetheHeitlerInteractor(
    const BetheHeitlerData&    shared,
    const ParticleTrackView&   particle,
    const Real3&               inc_direction,
    StackAllocator<Secondary>& allocate,
    const ElementView&         element)
    : shared_(shared)
    , inc_energy_(particle.energy().value())
    , inc_direction_(inc_direction)
    , allocate_(allocate)
    , element_(element)
{
    CELER_EXPECT(particle.particle_id() == shared_.gamma_id);
    CELER_EXPECT(inc_energy_.value() > 2 * shared_.electron_mass);

    epsilon0_ = shared_.electron_mass / inc_energy_.value();
    CELER_ENSURE(epsilon0_ < real_type(0.5));
}

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Pair-production using the Bethe-Heitler model.
 *
 * See section 6.5 of the Geant physics reference 10.6.
 */
template<class Engine>
CELER_FUNCTION Interaction BetheHeitlerInteractor::operator()(Engine& rng)
{
    // Allocate space for the pair-produced positron
    Secondary* positron = this->allocate_(1);
    if (positron == nullptr)
    {
        // Failed to allocate space for a secondary
        return Interaction::from_failure();
    }

    constexpr real_type half = 0.5;

    // If E_gamma < 2 MeV, rejection not needed -- just sample uniformly
    real_type epsilon;
    if (inc_energy_ < units::MevEnergy{2.0})
    {
        UniformRealDistribution<real_type> sample_eps(epsilon0_, half);
        epsilon = sample_eps(rng);
    }
    else
    {
        // Minimum (\epsilon = 1/2) and maximum (\epsilon = \epsilon1) values
        // of screening variable, \delta.
        const real_type delta_min = 4 * 136 / element_.cbrt_z() * epsilon0_;
        const real_type delta_max
            = std::exp((real_type(42.24) - element_.coulomb_correction())
                       / real_type(8.368))
              - real_type(0.952);
        CELER_ASSERT(delta_min <= delta_max);

        // Limits on epsilon
        const real_type epsilon1
            = half - half * std::sqrt(1 - delta_min / delta_max);
        const real_type epsilon_min = celeritas::max(epsilon0_, epsilon1);

        // Decide to choose f1, g1 or f2, g2 based on N1, N2 (factors from
        // corrected Bethe-Heitler cross section; c.f. Eq. 6.6 of Geant4
        // Physics Reference 10.6)
        const real_type       f10 = this->screening_phi1_aux(delta_min);
        const real_type       f20 = this->screening_phi2_aux(delta_min);
        BernoulliDistribution choose_f1g1(ipow<2>(half - epsilon_min) * f10,
                                          real_type(1.5) * f20);

        // Temporary sample values used in rejection
        real_type reject_threshold;
        do
        {
            if (choose_f1g1(rng))
            {
                // Used to sample from f1
                epsilon = half
                          - (half - epsilon_min)
                                * std::cbrt(generate_canonical(rng));
                CELER_ASSERT(epsilon >= epsilon_min && epsilon <= half);
                // Calculate delta from element atomic number and sampled
                // epsilon
                real_type delta = this->impact_parameter(epsilon);
                CELER_ASSERT(delta <= delta_max && delta >= delta_min);
                // Calculate g1 "rejection" function
                reject_threshold = this->screening_phi1_aux(delta) / f10;
                CELER_ASSERT(reject_threshold > 0 && reject_threshold <= 1);
            }
            else
            {
                // Used to sample from f2
                epsilon = epsilon_min
                          + (half - epsilon_min) * generate_canonical(rng);
                CELER_ASSERT(epsilon >= epsilon_min && epsilon <= half);
                // Calculate delta given the element atomic number and sampled
                // epsilon
                real_type delta = this->impact_parameter(epsilon);
                CELER_ASSERT(delta <= delta_max && delta >= delta_min);
                // Calculate g2 "rejection" function
                reject_threshold = this->screening_phi2_aux(delta) / f20;
                CELER_ASSERT(reject_threshold > 0 && reject_threshold <= 1);
            }
        } while (!BernoulliDistribution(reject_threshold)(rng));
    }

    // Construct interaction for change to primary (incident) particle (gamma)
    Interaction result  = Interaction::from_absorption();
    result.secondaries  = {positron, 1};
    Secondary* electron = &result.secondary;

    // Outgoing secondaries are electron and positron
    electron->particle_id = shared_.electron_id;
    positron->particle_id = shared_.positron_id;
    electron->energy = units::MevEnergy{(1 - epsilon) * inc_energy_.value()
                                        - shared_.electron_mass};
    positron->energy = units::MevEnergy{epsilon * inc_energy_.value()
                                        - shared_.electron_mass};
    // Select charges for child particles (e-, e+) randomly
    if (BernoulliDistribution(half)(rng))
    {
        trivial_swap(electron->energy, positron->energy);
    }

    // Sample secondary directions.
    // Note that momentum is not exactly conserved.
    UniformRealDistribution<real_type> sample_phi(0, 2 * constants::pi);
    real_type                          phi = sample_phi(rng);

    // Electron
    TsaiUrbanDistribution sample_electron_angle(
        electron->energy, MevMass{shared_.electron_mass});
    real_type cost      = sample_electron_angle(rng);
    electron->direction = rotate(from_spherical(cost, phi), inc_direction_);
    // Positron
    TsaiUrbanDistribution sample_positron_angle(
        positron->energy, MevMass{shared_.electron_mass});
    cost = sample_positron_angle(rng);
    positron->direction
        = rotate(from_spherical(cost, phi + constants::pi), inc_direction_);
    return result;
}

CELER_FUNCTION real_type
BetheHeitlerInteractor::impact_parameter(real_type eps) const
{
    return 136 / element_.cbrt_z() * epsilon0_ / (eps * (1 - eps));
}

CELER_FUNCTION real_type
BetheHeitlerInteractor::screening_phi1(real_type delta) const
{
    using R = real_type;
    return delta <= R(1.4)
               ? R(20.867) - R(3.242) * delta + R(0.625) * ipow<2>(delta)
               : R(21.12) - R(4.184) * std::log(delta + R(0.952));
}

CELER_FUNCTION
real_type BetheHeitlerInteractor::screening_phi2(real_type delta) const
{
    using R = real_type;
    return delta <= R(1.4)
               ? R(20.209) - R(1.930) * delta - R(0.086) * ipow<2>(delta)
               : R(21.12) - R(4.184) * std::log(delta + R(0.952));
}

CELER_FUNCTION real_type
BetheHeitlerInteractor::screening_phi1_aux(real_type delta) const
{
    // TODO: maybe assert instead of "clamp"? Not clear when this is negative
    return celeritas::clamp_to_nonneg(3 * this->screening_phi1(delta)
                                      - this->screening_phi2(delta)
                                      - element_.coulomb_correction());
}

CELER_FUNCTION real_type
BetheHeitlerInteractor::screening_phi2_aux(real_type delta) const
{
    // TODO: maybe assert instead of "clamp"? Not clear when this is negative
    using R = real_type;
    return celeritas::clamp_to_nonneg(R(1.5) * this->screening_phi1(delta)
                                      - R(0.5) * this->screening_phi2(delta)
                                      - element_.coulomb_correction());
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
