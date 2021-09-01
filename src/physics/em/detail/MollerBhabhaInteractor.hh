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
#include "physics/base/CutoffView.hh"
#include "base/StackAllocator.hh"
#include "physics/base/Secondary.hh"
#include "physics/base/Units.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Device data for creating an interactor.
 */
struct MollerBhabhaPointers
{
    //! Model ID
    ModelId model_id;

    //! ID of an electron
    ParticleId electron_id;
    //! ID of a positron
    ParticleId positron_id;
    //! Electron mass * c^2 [MeV]
    real_type electron_mass_c_sq;
    //! Model's mininum energy limit [MeV]
    static CELER_CONSTEXPR_FUNCTION real_type min_valid_energy()
    {
        return 1e-3;
    }
    //! Model's maximum energy limit [MeV]
    static CELER_CONSTEXPR_FUNCTION real_type max_valid_energy()
    {
        return 100e6;
    }

    //! Check whether the data is assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        return electron_id && positron_id && electron_mass_c_sq > 0;
    }
};

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
    //! Construct with shared and state data
    inline CELER_FUNCTION
    MollerBhabhaInteractor(const MollerBhabhaPointers& shared,
                           const ParticleTrackView&    particle,
                           const CutoffView&           cutoffs,
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
    // Secondary cutoff value for current particle and material
    real_type secondary_energy_cutoff_;
    // Allocate space for the secondary particle
    StackAllocator<Secondary>& allocate_;
    // Incident particle flag for selecting Moller or Bhabha scattering
    const bool inc_particle_is_electron_;
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

#include "MollerBhabhaInteractor.i.hh"
