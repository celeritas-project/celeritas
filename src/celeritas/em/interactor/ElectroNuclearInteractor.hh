//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/interactor/ElectroNuclearInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "celeritas/em/data/ElectroNuclearData.hh"
#include "celeritas/phys/Interaction.hh"
#include "celeritas/phys/ParticleTrackView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Handle the electro-nuclear interaction using G4ElectroVDNuclearModel.
 *
 * The electro-nuclear interaction requires hadronic models for the final
 * state generation, as described in section 45.2 of the Geant4 physics manual.
 * When the electro-nuclear process is selected, the electromagnetic vertex
 * of the electro-nucleus reaction is computed and the virtual photon is
 * generated. The status is set to onload::electro_nuclear and the post step
 * action of the converted real photon will be handled by Geant4, while the
 * primary electron or position continues to be tracked.
 */
class ElectroNuclearInteractor
{
  public:
    // Construct from shared and state data
    inline CELER_FUNCTION
    ElectroNuclearInteractor(NativeCRef<ElectroNuclearData> const& shared,
                             ParticleTrackView const& particle);

    // Sample an interaction
    inline CELER_FUNCTION Interaction operator()();
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data, and a target nucleus.
 */
CELER_FUNCTION
ElectroNuclearInteractor::ElectroNuclearInteractor(
    NativeCRef<ElectroNuclearData> const& shared,
    ParticleTrackView const& particle)
{
    CELER_EXPECT(particle.particle_id() == shared.scalars.electron_id
                 || particle.particle_id() == shared.scalars.positron_id);
}

//---------------------------------------------------------------------------//
/*!
 * Onload the electro-nuclear interaction.
 */
CELER_FUNCTION Interaction ElectroNuclearInteractor::operator()()
{
    return Interaction::from_onloaded();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
