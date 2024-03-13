//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/neutron/interactor/NeutronInelasticInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/mat/IsotopeView.hh"
#include "celeritas/neutron/data/NeutronInelasticData.hh"
#include "celeritas/phys/Interaction.hh"
#include "celeritas/phys/ParticleTrackView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Perform neutron inelastic interaction based on the Bertini cascade model.
 *
 * \note This performs the sampling procedure as in G4CascadeInterface, as
 * documented in section 24 of the Geant4 Physics Reference (release 11.2).
 */
class NeutronInelasticInteractor
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
    NeutronInelasticInteractor(NeutronInelasticRef const& shared,
                               ParticleTrackView const& particle,
                               Real3 const& inc_direction,
                               IsotopeView const& target);

    // Sample an interaction with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng);

  private:
    //// DATA ////

    // Constant shared data
    NeutronInelasticRef const& shared_;
    // Incident neutron direction
    Real3 const& inc_direction_;
    // Target nucleus
    IsotopeView const& target_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data, and a target nucleus.
 */
CELER_FUNCTION
NeutronInelasticInteractor::NeutronInelasticInteractor(
    NeutronInelasticRef const& shared,
    ParticleTrackView const& particle,
    Real3 const& inc_direction,
    IsotopeView const& target)
    : shared_(shared), inc_direction_(inc_direction), target_(target)
{
    CELER_EXPECT(particle.particle_id() == shared_.ids.neutron);
}
//---------------------------------------------------------------------------//
/*!
 * Sample the final state of the neutron-nucleus inelastic interaction.
 */
template<class Engine>
CELER_FUNCTION Interaction NeutronInelasticInteractor::operator()(Engine&)
{
    // Dummy for now
    Interaction result;
    CELER_ENSURE(result.action == Interaction::Action::absorbed);

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas