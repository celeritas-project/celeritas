//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/interactor/detail/BremFinalStateHelper.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/Quantities.hh"
#include "celeritas/em/distribution/TsaiUrbanDistribution.hh"
#include "celeritas/phys/InteractionUtils.hh"
#include "celeritas/phys/Secondary.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Sample the angular distribution of photon from e+/e- Bremsstrahlung.
 */
class BremFinalStateHelper
{
  public:
    //!@{
    //! \name Type aliases
    using Energy = units::MevEnergy;
    using Mass = units::MevMass;
    using Momentum = units::MevMomentum;
    //!@}

  public:
    // Construct from data
    inline CELER_FUNCTION BremFinalStateHelper(Energy inc_energy,
                                               Real3 const& inc_direction,
                                               Momentum inc_momentum,
                                               Mass inc_mass,
                                               ParticleId gamma_id,
                                               Energy gamma_energy,
                                               Secondary* secondary);

    // Update the final state for the given RNG and the photon energy
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng);

  private:
    // Incident particle direction
    Real3 const& inc_direction_;
    // Incident particle momentum
    Momentum inc_momentum_;
    // Exiting energy
    Energy exit_energy_;
    // Bremsstrahlung photon id
    ParticleId gamma_id_;
    // Exiting gamma energy
    Energy gamma_energy_;
    // Allocated secondary gamma
    Secondary* secondary_;
    // Secondary angular distribution
    TsaiUrbanDistribution sample_polar_angle_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from incident particle and exiting gamma data.
 */
CELER_FUNCTION
BremFinalStateHelper::BremFinalStateHelper(Energy inc_energy,
                                           Real3 const& inc_direction,
                                           Momentum inc_momentum,
                                           Mass inc_mass,
                                           ParticleId gamma_id,
                                           Energy gamma_energy,
                                           Secondary* secondary)
    : inc_direction_(inc_direction)
    , inc_momentum_(inc_momentum)
    , exit_energy_{inc_energy - gamma_energy}
    , gamma_id_(gamma_id)
    , gamma_energy_{gamma_energy}
    , secondary_{secondary}
    , sample_polar_angle_{inc_energy, inc_mass}
{
    CELER_EXPECT(secondary_);
}

//---------------------------------------------------------------------------//
/*!
 * Update the final state of the primary particle and the secondary photon.
 */
template<class Engine>
CELER_FUNCTION Interaction BremFinalStateHelper::operator()(Engine& rng)
{
    // Generate exiting gamma direction from isotropic azimuthal angle and
    // TsaiUrbanDistribution for polar angle (based on G4ModifiedTsai)
    secondary_->direction = ExitingDirectionSampler{sample_polar_angle_(rng),
                                                    inc_direction_}(rng);
    secondary_->particle_id = gamma_id_;
    secondary_->energy = gamma_energy_;

    // Construct interaction for change to parent (incoming) particle
    Interaction result;
    result.energy = exit_energy_;
    result.direction = calc_exiting_direction(
        {inc_momentum_.value(), inc_direction_},
        {gamma_energy_.value(), secondary_->direction});
    result.secondaries = {secondary_, 1};

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
