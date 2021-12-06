//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file KleinNishinaInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Types.hh"
#include "physics/base/Interaction.hh"
#include "physics/base/ParticleTrackView.hh"
#include "base/StackAllocator.hh"
#include "physics/base/Secondary.hh"
#include "physics/base/Units.hh"
#include "KleinNishinaData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Perform Compton scattering, neglecting atomic binding energy.
 *
 * This is a model for the discrete Compton inelastic scattering process. Given
 * an incident gamma, it adds a single secondary (electron) to the secondary
 * stack and returns an interaction for the change to the incident gamma
 * direction and energy. No cutoffs are performed for the incident energy or
 * the exiting gamma energy. A secondary production cutoff is applied to the
 * outgoing electron.
 *
 * \note This performs the same sampling routine as in Geant4's
 *  G4KleinNishinaCompton, as documented in section 6.4.2 of the Geant4 Physics
 *  Reference (release 10.6).
 */
class KleinNishinaInteractor
{
  public:
    // Construct from shared and state data
    inline CELER_FUNCTION
    KleinNishinaInteractor(const KleinNishinaData&    shared,
                           const ParticleTrackView&   particle,
                           const Real3&               inc_direction,
                           StackAllocator<Secondary>& allocate);

    // Sample an interaction with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng);

    //! Energy threshold for secondary production [MeV]
    static CELER_CONSTEXPR_FUNCTION units::MevEnergy secondary_cutoff()
    {
        return units::MevEnergy{1e-4};
    }

  private:
    // Constant data
    const KleinNishinaData& shared_;
    // Incident gamma energy
    const units::MevEnergy inc_energy_;
    // Incident direction
    const Real3& inc_direction_;
    // Allocate space for a secondary particle
    StackAllocator<Secondary>& allocate_;
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

#include "KleinNishinaInteractor.i.hh"
