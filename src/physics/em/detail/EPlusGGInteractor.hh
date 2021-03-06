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
#include "base/StackAllocator.hh"
#include "physics/base/Secondary.hh"
#include "physics/base/Units.hh"
#include "EPlusGG.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Annihilate a positron to create two gammas.
 *
 * This is a model for the discrete positron-electron annihilation process
 * which simulates the in-flight annihilation of a positron with an atomic
 * electron and produces into two photons. It is assumed that the atomic
 * electron is initially free and at rest.
 *
 * \note This performs the same sampling routine as in Geant4's
 * G4eeToTwoGammaModel class, as documented in section 10.3 of the Geant4
 * Physics Reference (release 10.6). The maximum energy for the model is
 * specified in Geant4.
 */
class EPlusGGInteractor
{
  public:
    // Construct with shared and state data
    inline CELER_FUNCTION
    EPlusGGInteractor(const EPlusGGPointers&     shared,
                      const ParticleTrackView&   particle,
                      const Real3&               inc_direction,
                      StackAllocator<Secondary>& allocate);

    // Sample an interaction with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng);

  private:
    // Shared constant physics properties
    const EPlusGGPointers& shared_;
    // Incident positron energy
    const real_type inc_energy_;
    // Incident direction
    const Real3& inc_direction_;
    // Allocate space for secondary particles (two photons)
    StackAllocator<Secondary>& allocate_;
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

#include "EPlusGGInteractor.i.hh"
