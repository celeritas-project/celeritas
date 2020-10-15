//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file BremRelInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/NumericLimits.hh"
#include "base/Macros.hh"
#include "base/Types.hh"
#include "physics/base/Interaction.hh"
#include "physics/base/MaterialTrackView.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/SecondaryAllocatorView.hh"
#include "physics/base/Secondary.hh"
#include "physics/base/Units.hh"
#include "BremRelInteractorPointers.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Relativistic bremsstrahlung interaction for electrons.
 *
 * This is a model for the bremsstrahlung process with relativistic effects and
 * two higher-order corrections, Landau-Pomeranchuk-Migdal (multiple
 * scattering) and Ter-Mikaelian (dialectric suppression).
 *
 * TODO:
 * - Geant4 offers an option to disable LPM.
 *
 * \note This performs the same sampling routine as in Geant4's
 * G4eBremsstrahlungRelModel class, as documented in section 10.2.2 of the
 * Geant4 Physics Reference (release 10.6).
 */
class BremRelInteractor
{
  public:
    //@{
    using MevEnergy = units::MevEnergy;
    //@}
  public:
    // Construct with shared and state data
    inline CELER_FUNCTION
    BremRelInteractor(const BremRelInteractorPointers& shared,
                      const ParticleTrackView&         particle,
                      const MaterialTrackView&         mat,
                      const Real3&                     inc_direction,
                      SecondaryAllocatorView&          allocate);

    // Sample an interaction with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng);

    // >>> COMMON PROPERTIES

    //! Minimum incident energy for this model to be valid
    static CELER_CONSTEXPR_FUNCTION units::MevEnergy min_incident_energy()
    {
        return MevEnergy{1};
    }

    //! Maximum incident energy for this model to be valid
    static CELER_CONSTEXPR_FUNCTION units::MevEnergy max_incident_energy()
    {
        return MevEnergy{celeritas::numeric_limits<real_type>::infinity()};
    }

  private:
    // Characteristic energy for LPM effects [MeV]
    CELER_FUNCTION real_type lpm_energy() const
    {
        return shared_.lpm_constant * mat_.radiation_length_tsai();
    }

  private:
    // Shared constant physics properties
    const BremRelInteractorPointers& shared_;
    // Allocate space for one or more secondary particles
    SecondaryAllocatorView& allocate_;

    //@{
    //! Material-dependent properties
    // Material reference
    const MaterialTrackView& mat_;
    // Density correction for this material [MeV^2]
    real_type density_corr_;
    //@}

    //@{
    //! State-dependent properties
    // Incident gamma energy
    const units::MevEnergy inc_energy_;
    // Incident direction
    const Real3& inc_direction_;
    // Whether LPM correction is used for the material and energy
    bool use_lpm_;
    //@}
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "BremRelInteractor.i.hh"
