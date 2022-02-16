//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file BremFinalStateHelper.hh
//---------------------------------------------------------------------------//
#pragma once

#include "physics/base/Units.hh"
#include "physics/base/Secondary.hh"
#include "random/distributions/UniformRealDistribution.hh"
#include "TsaiUrbanDistribution.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Sample the angular distribution of photon from e+/e- Bremsstrahlung.
 *
 */
class BremFinalStateHelper
{
  public:
    //!@{
    //! Type aliases
    using Energy   = units::MevEnergy;
    using Mass     = units::MevMass;
    using Momentum = units::MevMomentum;
    //!@}

  public:
    // Construct from data
    inline CELER_FUNCTION BremFinalStateHelper(const Energy&     inc_energy,
                                               const Real3&      inc_direction,
                                               const Momentum&   inc_momentum,
                                               const Mass&       inc_mass,
                                               const ParticleId& gamma_id);

    // Update the final state for the given RNG and the photon energy
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine&      rng,
                                                 const Energy gamma_energy);

  private:
    // Incident particle energy
    const Energy inc_energy_;
    // Incident particle direction
    const Real3& inc_direction_;
    // Incident particle momentum
    const Momentum inc_momentum_;
    // Incident particle mass
    const Mass inc_mass_;
    // Bremsstrahlung photon id
    const ParticleId gamma_id_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from incident particle and shared data.
 */
CELER_FUNCTION
BremFinalStateHelper::BremFinalStateHelper(const Energy&     inc_energy,
                                           const Real3&      inc_direction,
                                           const Momentum&   inc_momentum,
                                           const Mass&       inc_mass,
                                           const ParticleId& gamma_id)
    : inc_energy_(inc_energy)
    , inc_direction_(inc_direction)
    , inc_momentum_(inc_momentum)
    , inc_mass_(inc_mass)
    , gamma_id_(gamma_id)
{
}

//---------------------------------------------------------------------------//
/*!
 * Update the final state of the primary particle and the secondary photon
 */
template<class Engine>
CELER_FUNCTION Interaction
BremFinalStateHelper::operator()(Engine& rng, const Energy gamma_energy)
{
    // Construct interaction for change to parent (incoming) particle
    Interaction result;
    result.action = Action::scattered;
    result.energy
        = units::MevEnergy{inc_energy_.value() - gamma_energy.value()};
    result.secondary.particle_id = gamma_id_;
    result.secondary.energy      = gamma_energy;

    // Generate exiting gamma direction from isotropic azimuthal angle and
    // TsaiUrbanDistribution for polar angle (based on G4ModifiedTsai)
    UniformRealDistribution<real_type> sample_phi(0, 2 * constants::pi);
    TsaiUrbanDistribution sample_gamma_angle(inc_energy_, inc_mass_);
    real_type             cost = sample_gamma_angle(rng);
    result.secondary.direction
        = rotate(from_spherical(cost, sample_phi(rng)), inc_direction_);

    // Update parent particle direction
    for (unsigned int i : range(3))
    {
        real_type inc_momentum_i   = inc_momentum_.value() * inc_direction_[i];
        real_type gamma_momentum_i = result.secondary.energy.value()
                                     * result.secondary.direction[i];
        result.direction[i] = inc_momentum_i - gamma_momentum_i;
    }
    normalize_direction(&result.direction);

    return result;
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
