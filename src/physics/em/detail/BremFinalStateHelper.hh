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
                                                 const Energy gamma_energy,
                                                 Secondary*   secondaries);

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
} // namespace detail
} // namespace celeritas

#include "BremFinalStateHelper.i.hh"
