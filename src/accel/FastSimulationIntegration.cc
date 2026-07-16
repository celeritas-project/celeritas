//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/FastSimulationIntegration.cc
//---------------------------------------------------------------------------//
#include "FastSimulationIntegration.hh"

#include <G4Threading.hh>
#include <G4Version.hh>

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"

#include "ExceptionConverter.hh"

namespace celeritas
{

//---------------------------------------------------------------------------//
/*!
 * Access the public-facing integration singleton.
 */
FastSimulationIntegration& FastSimulationIntegration::Instance()
{
    static FastSimulationIntegration tmi;
    return tmi;
}

//---------------------------------------------------------------------------//
/*!
 * Verify fast simulation setup.
 */
void FastSimulationIntegration::verify_local_setup()
{
    CELER_VALIDATE(G4VERSION_NUMBER >= 1110,
                   << "the current version of Geant4 (" << G4VERSION_NUMBER
                   << ") is too old to support the fast simulation offload "
                      "interface (11.1 or higher is required)");

    // TODO: Check requested offload particles have G4 fast sim process/manager

    // TODO: Check requested regions have our FastSimulationModel attached
}

//---------------------------------------------------------------------------//
/*!
 * Only allow the singleton to construct.
 */
FastSimulationIntegration::FastSimulationIntegration() = default;

//---------------------------------------------------------------------------//
}  // namespace celeritas
