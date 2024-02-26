//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/neutron/interactor/ChipsNeutronElasticInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/ArrayOperators.hh"
#include "celeritas/Constants.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/mat/IsotopeView.hh"
#include "celeritas/neutron/data/NeutronElasticData.hh"
#include "celeritas/phys/FourVector.hh"
#include "celeritas/phys/Interaction.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/random/distribution/UniformRealDistribution.hh"

#include "detail/MomentumTransferSampler.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Perform neutron elastic scattering based on the CHIPS (Chiral invariant
 * phase space) model.
 *
 * \note This performs the sampling procedure as in G4HadronElastic,
 * G4ChipsElasticModel and G4ChipsNeutronElasticXS, as partly documented
 * in section 21.1.3 of the Geant4 Physics Reference (release 11.2).
 */
class ChipsNeutronElasticInteractor
{
  public:
    //!@{
    //! \name Type aliases
    using Energy = units::MevEnergy;
    using Mass = units::MevMass;
    using Momentum = units::MevMomentum;
    //!@}

  public:
    // Construct from shared and state data
    inline CELER_FUNCTION
    ChipsNeutronElasticInteractor(NeutronElasticRef const& shared,
                                  ParticleTrackView const& particle,
                                  Real3 const& inc_direction,
                                  IsotopeView const& target);

    // Sample an interaction with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng);

  private:
    //// TYPES ////

    using UniformRealDist = UniformRealDistribution<real_type>;

    //// DATA ////

    // Constant shared data
    NeutronElasticRef const& shared_;
    // Incident neutron direction
    Real3 const& inc_direction_;
    // Target nucleus
    IsotopeView const& target_;

    // Values of neutron mass (MevMass) and energy (MevEnergy)
    real_type neutron_mass_;
    real_type neutron_energy_;
    Momentum neutron_p_;

    // Sampler
    UniformRealDist sample_phi_;
    detail::MomentumTransferSampler sample_momentum_square_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data, and a target nucleus.
 */
CELER_FUNCTION ChipsNeutronElasticInteractor::ChipsNeutronElasticInteractor(
    NeutronElasticRef const& shared,
    ParticleTrackView const& particle,
    Real3 const& inc_direction,
    IsotopeView const& target)
    : shared_(shared)
    , inc_direction_(inc_direction)
    , target_(target)
    , neutron_mass_(value_as<Mass>(shared_.neutron_mass))
    , neutron_energy_(neutron_mass_ + value_as<Energy>(particle.energy()))
    , neutron_p_(particle.momentum())
    , sample_phi_(0, 2 * constants::pi)
    , sample_momentum_square_(shared_, target_, neutron_p_)
{
    CELER_EXPECT(particle.particle_id() == shared_.ids.neutron);
}
//---------------------------------------------------------------------------//
/*!
 * Sample the final state of the neutron-nucleus elastic scattering.
 */
template<class Engine>
CELER_FUNCTION Interaction ChipsNeutronElasticInteractor::operator()(Engine& rng)
{
    // Scattered neutron with respect to the axis of incident direction
    Interaction result;

    // The momentum magnitude in the c.m. frame
    real_type target_mass = value_as<Mass>(target_.nuclear_mass());

    real_type cm_p = value_as<units::MevMomentum>(neutron_p_)
                     / std::sqrt(1 + ipow<2>(neutron_mass_ / target_mass)
                                 + 2 * neutron_energy_ / target_mass);

    // Sample the scattered direction from the invariant momentum transfer
    // squared (\f$ -t = Q^{2} \f$) in the c.m. frame
    real_type cos_theta
        = 1 - real_type(0.5) * sample_momentum_square_(rng) / ipow<2>(cm_p);
    clamp(cos_theta, real_type{-1}, real_type{1});

    // Boost to the center of mass (c.m.) frame
    Real3 cm_mom = cm_p * from_spherical(cos_theta, sample_phi_(rng));
    FourVector nlv1(
        {cm_mom, std::sqrt(ipow<2>(cm_p) + ipow<2>(neutron_mass_))});

    FourVector lv({{0, 0, value_as<units::MevMomentum>(neutron_p_)},
                   neutron_energy_ + target_mass});
    boost(boost_vector(lv), &nlv1);

    result.direction
        = make_unit_vector(rotate(make_unit_vector(nlv1.mom), inc_direction_));

    // Kinetic energy of the scattered neutron and the recoiled nucleus
    lv.energy -= nlv1.energy;
    result.energy = Energy(nlv1.energy - neutron_mass_);
    real_type recoil_energy = clamp_to_nonneg(lv.energy - target_mass);
    result.energy_deposition = Energy{recoil_energy};

    // TODO: create a secondary ion(Z, N) if recoil_energy > recoil_threshold
    // with energy = recoil_energy and direction = lv.mom - nlv1.mom
    // Note: the tracking of the secondary ion is only needed when there is
    // a detail simulation of radiative decay for the recoiled nucleus.

    CELER_ENSURE(result.action == Interaction::Action::scattered);

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
