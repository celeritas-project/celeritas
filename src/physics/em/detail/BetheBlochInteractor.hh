//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file BetheBlochInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Types.hh"
#include "physics/base/Interaction.hh"
#include "physics/base/ParticleTrackView.hh"
#include "base/StackAllocator.hh"
#include "physics/base/Secondary.hh"
#include "physics/base/Units.hh"
#include "physics/material/ElementView.hh"
#include "physics/material/MaterialView.hh"
#include "BetheBloch.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Muon bremsstrahlung.
 *
 * \note This performs the same sampling routine as in Geant4's
 * G4MuBremsstrahlungModel class, as documented in section 11.2 
 * of the Geant4 Physics Reference (release 10.6).
 */
class BetheBlochInteractor
{
  public:
    // Construct with shared and state data
    inline CELER_FUNCTION
    BetheBlochInteractor(const BetheBlochInteractorPointers& shared,
                         const ParticleTrackView&            particle,
                         const Real3&                        inc_direction,
                         StackAllocator<Secondary>&          allocate,
                         MaterialView&                       material);

    // Sample an interaction with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng);

    //// COMMON PROPERTIES ////

    //! Minimum incident energy for this model to be valid
    static CELER_CONSTEXPR_FUNCTION units::MevEnergy min_incident_energy()
    {
        return units::MevEnergy{0.2}; 
    }

    //! Maximum incident energy for this model to be valid
    static CELER_CONSTEXPR_FUNCTION units::MevEnergy max_incident_energy()
    {
        return units::MevEnergy{1000};
    }

  private:
    template<class Engine>
    CELER_FUNCTION real_type sample_cos_theta(real_type gamma_energy, 
                                              Engine& rng);

    CELER_FUNCTION real_type differential_cross_section(real_type gamma_enrgy, 
                                                        ElementView element);

    // Shared constant physics properties
    const BetheBlochInteractorPointers& shared_;
    // Incident muon energy
    const units::MevEnergy inc_energy_;
    // Incident direction
    const Real3& inc_direction_;
    // Allocate space for one or more secondary particles
    StackAllocator<Secondary>& allocate_;
    // Material properties
    const MaterialView& material_;
    // Incident muon mass
    const units::MevMass inc_mass_;
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

#include "BetheBlochInteractor.i.hh"

