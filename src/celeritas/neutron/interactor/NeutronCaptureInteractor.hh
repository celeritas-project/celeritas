//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/neutron/interactor/NeutronCaptureInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/neutron/data/NeutronCaptureData.hh"
#include "celeritas/phys/Interaction.hh"
#include "celeritas/phys/ParticleTrackView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Handle the neutron capture interaction.
 *
 * When the neutron capture process is selected, the post step action will be
 * handled by Geant4.
 */
class NeutronCaptureInteractor
{
  public:
    // Construct from shared and state data
    inline CELER_FUNCTION
    NeutronCaptureInteractor(NativeCRef<NeutronCaptureData> const& shared,
                             ParticleTrackView const& particle);

    // Sample an interaction
    inline CELER_FUNCTION Interaction operator()();
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
CELER_FUNCTION
NeutronCaptureInteractor::NeutronCaptureInteractor(
    NativeCRef<NeutronCaptureData> const& shared,
    ParticleTrackView const& particle)
{
    CELER_EXPECT(particle.particle_id() == shared.neutron_id);
}
//---------------------------------------------------------------------------//
/*!
 * Onload the neutron capture interaction.
 */
CELER_FUNCTION Interaction NeutronCaptureInteractor::operator()()
{
    return Interaction::from_onloaded();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
