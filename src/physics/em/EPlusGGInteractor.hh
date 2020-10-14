//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file EPlusGGInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Types.hh"
#include "physics/base/Interaction.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/SecondaryAllocatorView.hh"
#include "physics/base/Secondary.hh"
#include "physics/base/Units.hh"
#include "EPlusGGInteractorPointers.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Brief class description.
 *
 * This is a model for XXXX process. Additional description
 *
 * \note This performs the same sampling routine as in Geant4's
 * XXXX class, as documented in section XXX of the Geant4 Physics
 * Reference (release 10.6).
 */
class EPlusGGInteractor
{
  public:
    // Construct with shared and state data
    inline CELER_FUNCTION
    EPlusGGInteractor(const EPlusGGInteractorPointers& shared,
                      const ParticleTrackView&         particle,
                      const Real3&                     inc_direction,
                      SecondaryAllocatorView&          allocate);

    // Sample an interaction with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng);

    // >>> COMMON PROPERTIES

    //! Minimum incident energy for this model to be valid
    static CELER_CONSTEXPR_FUNCTION units::MevEnergy min_incident_energy()
    {
        return units::MevEnergy{0}; // XXX
    }

    //! Maximum incident energy for this model to be valid
    static CELER_CONSTEXPR_FUNCTION units::MevEnergy max_incident_energy()
    {
        return units::MevEnergy{0}; // XXX
    }

  private:
    // Shared constant physics properties
    const EPlusGGInteractorPointers& shared_;
    // Incident gamma energy
    const units::MevEnergy inc_energy_;
    // Incident direction
    const Real3& inc_direction_;
    // Allocate space for one or more secondary particles
    SecondaryAllocatorView& allocate_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "EPlusGGInteractor.i.hh"
