//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/interactor/MuBremsstrahlungInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/StackAllocator.hh"
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/Constants.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/em/data/MuBremsstrahlungData.hh"
#include "celeritas/em/distribution/TsaiUrbanDistribution.hh"
#include "celeritas/em/xs/MuBremsDiffXsCalculator.hh"
#include "celeritas/mat/ElementView.hh"
#include "celeritas/mat/MaterialView.hh"
#include "celeritas/phys/CutoffView.hh"
#include "celeritas/phys/Interaction.hh"
#include "celeritas/phys/InteractionUtils.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/phys/Secondary.hh"
#include "celeritas/random/distribution/BernoulliDistribution.hh"

#include "detail/BremFinalStateHelper.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Perform muon bremsstrahlung interaction.
 *
 * This is a model for the Bremsstrahlung process for muons. Given an incident
 * muon, the class computes the change to the incident muon direction and
 * energy, and it adds a single secondary gamma to the secondary stack.
 *
 * \note This performs the same sampling routine as in Geant4's
 * G4MuBremsstrahlungModel class, as documented in section 11.2
 * of the Geant4 Physics Reference (release 10.6).
 */
class MuBremsstrahlungInteractor
{
    //!@{
    //! \name Type aliases
    using Energy = units::MevEnergy;
    using Mass = units::MevMass;
    using Momentum = units::MevMomentum;
    //!@}

  public:
    // Construct with shared and state data
    inline CELER_FUNCTION
    MuBremsstrahlungInteractor(MuBremsstrahlungData const& shared,
                               ParticleTrackView const& particle,
                               Real3 const& inc_direction,
                               CutoffView const& cutoffs,
                               StackAllocator<Secondary>& allocate,
                               MaterialView const& material,
                               ElementComponentId elcomp_id);

    // Sample an interaction with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng);

  private:
    //// DATA ////

    // Shared constant physics properties
    MuBremsstrahlungData const& shared_;
    // Incident direction
    Real3 const& inc_direction_;
    // Allocate space for one or more secondary particles
    StackAllocator<Secondary>& allocate_;
    // Element properties
    ElementView const element_;
    // Incident particle
    ParticleTrackView const& particle_;
    // Differential cross section calculator
    MuBremsDiffXsCalculator calc_dcs_;
    // Ratio of gamma production cutoff to minimum energy cutoff
    real_type xmin_;
    // Ratio of incident energy to gamma production cutoff
    real_type xmax_;
    // Envelope distribution for rejection sampling of gamma energy
    real_type envelope_;

    //// CONSTANTS ////

    //! Minimum allowed secondary cutoff energy [MeV]
    static CELER_CONSTEXPR_FUNCTION Energy min_cutoff_energy()
    {
        return units::MevEnergy{9e-4};  //!< 0.9 keV
    }

    //// HELPER FUNCTIONS ////

    template<class Engine>
    inline CELER_FUNCTION real_type sample_cos_theta(real_type gamma_energy,
                                                     Engine& rng) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
CELER_FUNCTION MuBremsstrahlungInteractor::MuBremsstrahlungInteractor(
    MuBremsstrahlungData const& shared,
    ParticleTrackView const& particle,
    Real3 const& inc_direction,
    CutoffView const& cutoffs,
    StackAllocator<Secondary>& allocate,
    MaterialView const& material,
    ElementComponentId elcomp_id)
    : shared_(shared)
    , inc_direction_(inc_direction)
    , allocate_(allocate)
    , element_(material.make_element_view(elcomp_id))
    , particle_(particle)
    , calc_dcs_(
          element_, particle.energy(), particle.mass(), shared.electron_mass)
    , xmin_(std::log(value_as<Energy>(cutoffs.energy(shared.gamma))
                     / value_as<Energy>(min_cutoff_energy())))
    , xmax_(std::log(value_as<Energy>(particle_.energy())
                     / value_as<Energy>(cutoffs.energy(shared.gamma))))
    , envelope_(value_as<Energy>(cutoffs.energy(shared.gamma))
                * calc_dcs_(cutoffs.energy(shared.gamma)))
{
    CELER_EXPECT(particle.particle_id() == shared_.mu_minus
                 || particle.particle_id() == shared_.mu_plus);
    CELER_EXPECT(cutoffs.energy(shared.gamma) >= min_cutoff_energy());
    CELER_EXPECT(particle_.energy() > cutoffs.energy(shared.gamma));
}

//---------------------------------------------------------------------------//
/*!
 * Sample using the muon bremsstrahlung model.
 */
template<class Engine>
CELER_FUNCTION Interaction MuBremsstrahlungInteractor::operator()(Engine& rng)
{
    // Allocate space for gamma
    Secondary* secondary = allocate_(1);
    if (secondary == nullptr)
    {
        // Failed to allocate space for a secondary
        return Interaction::from_failure();
    }

    // Sample the energy transfer
    real_type gamma_energy;
    do
    {
        gamma_energy = value_as<Energy>(min_cutoff_energy())
                       * std::exp(xmin_ + xmax_ * generate_canonical(rng));
    } while (!BernoulliDistribution(
        gamma_energy * calc_dcs_(Energy{gamma_energy}) / envelope_)(rng));

    // Update kinematics of the final state and return this interaction
    return detail::BremFinalStateHelper(
        particle_.energy(),
        inc_direction_,
        particle_.momentum(),
        shared_.gamma,
        Energy{gamma_energy},
        this->sample_cos_theta(gamma_energy, rng),
        secondary)(rng);
}

//---------------------------------------------------------------------------//
/*!
 * Sample cosine of the angle between incident and secondary particles.
 */
template<class Engine>
CELER_FUNCTION real_type MuBremsstrahlungInteractor::sample_cos_theta(
    real_type gamma_energy, Engine& rng) const
{
    real_type const gamma = particle_.lorentz_factor();
    real_type const r_max
        = gamma * constants::pi * real_type(0.5)
          * min(real_type(1.0),
                gamma * value_as<Mass>(particle_.mass()) / gamma_energy - 1);
    real_type const r_max_sq = ipow<2>(r_max);

    real_type const a = generate_canonical(rng) * r_max_sq / (1 + r_max_sq);

    return std::cos(std::sqrt(a / (1 - a)) / gamma);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
