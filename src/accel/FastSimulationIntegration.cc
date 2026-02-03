//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/FastSimulationIntegration.cc
//---------------------------------------------------------------------------//
#include "FastSimulationIntegration.hh"

#include <G4Threading.hh>

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
    // TODO: Loop through regions to ensure at least one has a
    // celeritas::FastSimulationModel
}

//---------------------------------------------------------------------------//
/*!
 * Only allow the singleton to construct.
 */
FastSimulationIntegration::FastSimulationIntegration() = default;

//---------------------------------------------------------------------------//
}  // namespace celeritas
