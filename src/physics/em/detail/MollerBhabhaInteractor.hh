//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MollerBhabhaInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "base/Macros.hh"
#include "base/Types.hh"

#include "physics/base/Interaction.hh"
#include "physics/base/ParticleTrackView.hh"
#include "base/StackAllocator.hh"
#include "physics/base/Secondary.hh"
#include "physics/base/Units.hh"
#include "physics/material/ElementView.hh"

#include "MollerBhabha.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Perform Moller (e-e-) and Bhabha (e+e-) scattering.
 *
 * This is a model for both Moller and Bhabha scattering processes.
 *
 * \note This performs the same sampling routine as in Geant4's
 * G4MollerBhabhaModel class, as documented in section 10.1.4 of the Geant4
 * Physics Reference (release 10.6).
 */
class MollerBhabhaInteractor
{
  public:
    // Construct with shared and state data
    inline CELER_FUNCTION
    MollerBhabhaInteractor(const MollerBhabhaPointers& shared,
                           const ParticleTrackView&    particle,
                           const Real3&                inc_direction,
                           StackAllocator<Secondary>&  allocate);

    // Sample an interaction with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng);

  private:
    // Shared constant physics properties
    const MollerBhabhaPointers& shared_;
    // Incident energy [MeV]
    const real_type inc_energy_;
    // Incident momentum [MeV]
    const real_type inc_momentum_;
    // Incident direction
    const Real3& inc_direction_;
    // Allocate space for one or more secondary particles
    StackAllocator<Secondary>& allocate_;
    // Incident particle flag for selecting Moller or Bhabha scattering
    bool inc_particle_is_electron_;
}; // namespace MollerBhabhaInteractor

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

#include "MollerBhabhaInteractor.i.hh"
