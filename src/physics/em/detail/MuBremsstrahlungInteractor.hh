//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MuBremsstrahlungInteractor.hh
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

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Device data for creating an interactor.
 */
struct MuBremsstrahlungPointers
{
    //! Model ID
    ModelId model_id;
    //! ID of a gamma
    ParticleId gamma_id;
    //! ID of a muon
    ParticleId mu_minus_id;
    //! ID of a muon
    ParticleId mu_plus_id;
    //! Electron mass [MeV / c^2]
    units::MevMass electron_mass;

    //! Minimum incident energy for this model to be valid
    static CELER_CONSTEXPR_FUNCTION units::MevEnergy min_incident_energy()
    {
        return units::MevEnergy{1e3};
    }

    //! Maximum incident energy for this model to be valid
    static CELER_CONSTEXPR_FUNCTION units::MevEnergy max_incident_energy()
    {
        return units::MevEnergy{1e7};
    }

    //! Check whether the data is assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        return model_id && gamma_id && mu_minus_id && mu_plus_id
               && electron_mass.value() > 0;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Muon bremsstrahlung.
 *
 * This is a model for the Bremsstrahlung process for muons. Given an incident
 * muon, the class computes the change to the incident muon direction and
 * energy, and it adds a single secondary gamma to the secondary stack.
 *
 * \note This performs the same sampling routine as in Geant4's
 * G4MuBremsstrahlungModel class, as documented in section 11.2
 * of the Geant4 Physics Reference (release 10.6).
 */
class MuBremsstrahlungInteractor
{
  public:
    // Construct with shared and state data
    inline CELER_FUNCTION
    MuBremsstrahlungInteractor(const MuBremsstrahlungPointers& shared,
                               const ParticleTrackView&        particle,
                               const Real3&                    inc_direction,
                               StackAllocator<Secondary>&      allocate,
                               const MaterialView&             material,
                               ElementComponentId              elcomp_id);

    // Sample an interaction with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng);

  private:
    template<class Engine>
    inline CELER_FUNCTION real_type sample_cos_theta(real_type gamma_energy,
                                                     Engine&   rng) const;

    inline CELER_FUNCTION real_type
    differential_cross_section(real_type gamma_energy) const;

    // Shared constant physics properties
    const MuBremsstrahlungPointers& shared_;
    // Incident direction
    const Real3& inc_direction_;
    // Allocate space for one or more secondary particles
    StackAllocator<Secondary>& allocate_;
    // Element properties
    const ElementView element_;
    // Incident particle
    const ParticleTrackView& particle_;
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

#include "MuBremsstrahlungInteractor.i.hh"
