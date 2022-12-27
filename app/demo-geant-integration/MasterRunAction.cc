//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/MasterRunAction.cc
//---------------------------------------------------------------------------//
#include "MasterRunAction.hh"

#include <G4Threading.hh>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/io/Logger.hh"
#include "accel/ExceptionConverter.hh"

namespace demo_geant
{
//---------------------------------------------------------------------------//
/*!
 * Construct with Celeritas setup options and shared data.
 */
MasterRunAction::MasterRunAction(SPConstOptions options, SPParams params)
    : options_(options), params_(params)
{
    CELER_EXPECT(options_);
    CELER_EXPECT(params_);
}

//---------------------------------------------------------------------------//
/*!
 * Initialize Celeritas.
 */
void MasterRunAction::BeginOfRunAction(const G4Run* run)
{
    CELER_EXPECT(run);

    celeritas::ExceptionConverter call_g4exception{"celer0001"};
    CELER_TRY_ELSE(
        {
            // Initialize shared data
            params_->Initialize(*options_);
            CELER_ASSERT(*params_);
        },
        call_g4exception);
}

//---------------------------------------------------------------------------//
/*!
 * Finalize Celeritas.
 */
void MasterRunAction::EndOfRunAction(const G4Run*)
{
    CELER_EXPECT(G4Threading::IsMasterThread());
    CELER_LOG_LOCAL(status) << "Finalizing Celeritas";

    // Clear shared data and write if master thread
    params_->Finalize();
}

//---------------------------------------------------------------------------//
} // namespace demo_geant
