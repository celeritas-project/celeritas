//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/interactor/GammaNuclearInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "celeritas/em/data/GammaNuclearData.hh"
#include "celeritas/phys/Interaction.hh"
#include "celeritas/phys/ParticleTrackView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Handle the gamma-nuclear interaction.
 *
 * The gamma-nuclear interaction requires hadronic models for the final state
 * generation in Geant4 physics manual section 44.2. When the gamma-nuclear
 * process is selected, the status is set to onload::gamma_nuclear and the
 * post step action will be handled by Geant4.
 */
class GammaNuclearInteractor
{
  public:
    // Construct from shared and state data
    inline CELER_FUNCTION
    GammaNuclearInteractor(NativeCRef<GammaNuclearData> const& shared,
                           ParticleTrackView const& particle);

    // Sample an interaction
    inline CELER_FUNCTION Interaction operator()();
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
/*!
 * Construct with shared and state data, and a target nucleus.
 */
CELER_FUNCTION
GammaNuclearInteractor::GammaNuclearInteractor(
    NativeCRef<GammaNuclearData> const& shared,
    ParticleTrackView const& particle)
{
    CELER_EXPECT(particle.particle_id() == shared.scalars.gamma_id);
}

//---------------------------------------------------------------------------//
/*!
 * Onload the gamma-nuclear interaction.
 */
CELER_FUNCTION Interaction GammaNuclearInteractor::operator()()
{
    return Interaction::from_onloaded();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
