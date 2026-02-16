//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ddceler/CelerOpticalOffload.hh
//---------------------------------------------------------------------------//
#pragma once

#include <DDG4/Geant4SteppingAction.h>

namespace celeritas
{
namespace dd
{
//---------------------------------------------------------------------------//
/*!
 * DDG4 stepping action for optical distribution offloading to Celeritas.
 *
 * This action intercepts Cherenkov and scintillation photon generation in
 * Geant4 and offloads the distribution data to Celeritas for GPU tracking.
 *
 * The optical physics in Geant4 must be configured to *not* stack photons
 * (stack_photons = false) so that photon counts are calculated but photons
 * are not created in Geant4.
 */
class CelerOpticalOffload final : public dd4hep::sim::Geant4SteppingAction
{
  public:
    // Standard constructor
    CelerOpticalOffload(dd4hep::sim::Geant4Context* ctxt,
                        std::string const& name);

    // Delete copy/move
    DDG4_DEFINE_ACTION_CONSTRUCTORS(CelerOpticalOffload);

    // Stepping action callback
    virtual void operator()(G4Step const* step, G4SteppingManager* mgr) final;
};

//---------------------------------------------------------------------------//
}  // namespace dd
}  // namespace celeritas
