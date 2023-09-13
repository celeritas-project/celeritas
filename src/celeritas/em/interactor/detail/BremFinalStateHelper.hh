//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/interactor/detail/BremFinalStateHelper.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/Quantities.hh"
#include "celeritas/em/distribution/TsaiUrbanDistribution.hh"
#include "celeritas/phys/Secondary.hh"
#include "celeritas/random/distribution/UniformRealDistribution.hh"

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
    inline CELER_FUNCTION BremFinalStateHelper(Energy const& inc_energy,
                                               Real3 const& inc_direction,
                                               Momentum const& inc_momentum,
                                               Mass const& inc_mass,
                                               ParticleId const& gamma_id,
                                               const Energy gamma_energy,
                                               Secondary* secondaries);

    // Update the final state for the given RNG and the photon energy
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng);

  private:
    // Incident particle direction
    Real3 const& inc_direction_;
    // Incident particle momentum
    const Momentum inc_momentum_;
    // Exiting energy
    const Energy exit_energy_;
    // Bremsstrahlung photon id
    const ParticleId gamma_id_;
    // Exiting gamma energy
    const Energy gamma_energy_;
    // Allocated secondary gamma
    Secondary* secondary_;
    // Incident particle mass
    TsaiUrbanDistribution sample_polar_angle_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from incident particle and exiting gamma data.
 */
CELER_FUNCTION
BremFinalStateHelper::BremFinalStateHelper(Energy const& inc_energy,
                                           Real3 const& inc_direction,
                                           Momentum const& inc_momentum,
                                           Mass const& inc_mass,
                                           ParticleId const& gamma_id,
                                           const Energy gamma_energy,
                                           Secondary* secondaries)
    : inc_direction_(inc_direction)
    , inc_momentum_(inc_momentum)
    , exit_energy_{inc_energy.value() - gamma_energy.value()}
    , gamma_id_(gamma_id)
    , gamma_energy_{gamma_energy}
    , secondary_{secondaries}
    , sample_polar_angle_{inc_energy, inc_mass}
{
    CELER_EXPECT(secondary_);
}

//---------------------------------------------------------------------------//
/*!
 * Update the final state of the primary particle and the secondary photon
 */
template<class Engine>
CELER_FUNCTION Interaction BremFinalStateHelper::operator()(Engine& rng)
{
    // Construct interaction for change to parent (incoming) particle
    Interaction result;
    result.energy = exit_energy_;

    // Generate exiting gamma direction from isotropic azimuthal angle and
    // TsaiUrbanDistribution for polar angle (based on G4ModifiedTsai)
    UniformRealDistribution<real_type> sample_phi(0, 2 * constants::pi);
    real_type cost = sample_polar_angle_(rng);
    secondary_->direction
        = rotate(from_spherical(cost, sample_phi(rng)), inc_direction_);

    // Update parent particle direction
    for (int i = 0; i < 3; ++i)
    {
        real_type inc_momentum_i = inc_momentum_.value() * inc_direction_[i];
        real_type gamma_momentum_i = gamma_energy_.value()
                                     * secondary_->direction[i];
        result.direction[i] = inc_momentum_i - gamma_momentum_i;
    }
    normalize_direction(&result.direction);

    secondary_->particle_id = gamma_id_;
    secondary_->energy = gamma_energy_;
    result.secondaries = {secondary_, 1};

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
