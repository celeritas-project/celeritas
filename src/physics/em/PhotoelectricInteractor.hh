//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PhotoelectricInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Algorithms.hh"
#include "base/Macros.hh"
#include "base/Types.hh"
#include "physics/base/Interaction.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/SecondaryAllocatorView.hh"
#include "physics/base/Secondary.hh"
#include "physics/base/Units.hh"
#include "PhotoelectricInteractorPointers.hh"
#include "PhotoelectricMicroXsCalculator.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Livermore model for the photoelectric effect.
 *
 * \note This performs the same sampling routine as in Geant4's
 * G4LivermorePhotoElectricModel class, as documented in section 6.3.5 of the
 * Geant4 Physics Reference (release 10.6). Below 5 keV, tabulated subshell
 * cross sections are used. Above 5 keV, EPICS2014 subshell cross sections are
 * parameterized in two different energy intervals (which depend on atomic
 * number and K-shell binding energy). The angle of the emitted photoelectron
 * is sampled from the Sauter-Gavrila distribution.
 */
class PhotoelectricInteractor
{
  public:
    //!@{
    //! Type aliases
    using MevEnergy = units::MevEnergy;
    //!@}

  public:
    // Construct with shared and state data
    inline CELER_FUNCTION
    PhotoelectricInteractor(const PhotoelectricInteractorPointers& shared,
                            const ParticleTrackView&               particle,
                            const Real3&            inc_direction,
                            SecondaryAllocatorView& allocate);

    // Sample an interaction with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine&      rng,
                                                 ElementDefId el_id);

    //// COMMON PROPERTIES ////

    //! Minimum incident energy for this model to be valid
    static CELER_CONSTEXPR_FUNCTION MevEnergy min_incident_energy()
    {
        return MevEnergy{0};
    }

    //! Maximum incident energy for this model to be valid
    static CELER_CONSTEXPR_FUNCTION MevEnergy max_incident_energy()
    {
        return MevEnergy{celeritas::numeric_limits<real_type>::infinity()};
    }

  private:
    // Shared constant physics properties
    const PhotoelectricInteractorPointers& shared_;
    // Incident direction
    const Real3& inc_direction_;
    // Incident gamma energy
    const MevEnergy inc_energy_;
    // Allocate space for one or more secondary particles
    SecondaryAllocatorView& allocate_;
    // Microscopic cross section calculator
    PhotoelectricMicroXsCalculator calc_micro_xs_;
    // Reciprocal of the energy
    real_type inv_energy_;

    //// HELPER FUNCTIONS ////

    // Sample the direction of the emitted photoelectron
    template<class Engine>
    inline CELER_FUNCTION Real3 sample_direction(Engine& rng) const;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "PhotoelectricInteractor.i.hh"
