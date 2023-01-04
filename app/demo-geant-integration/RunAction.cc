//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/RunAction.cc
//---------------------------------------------------------------------------//
#include "RunAction.hh"

#include <G4Threading.hh>

#include "celeritas_config.h"
#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/io/Logger.hh"
#include "accel/ExceptionConverter.hh"

#include "GlobalSetup.hh"
#include "NoFieldAlongStepFactory.hh"

namespace demo_geant
{
//---------------------------------------------------------------------------//
/*!
 * Construct with Celeritas setup options and shared data.
 */
RunAction::RunAction(SPConstOptions options,
                     SPParams       params,
                     SPTransporter  transport)
    : options_(options), params_(params), transport_(transport)
{
    CELER_EXPECT(options_);
    CELER_EXPECT(params_);
    CELER_EXPECT(transport_);
}

//---------------------------------------------------------------------------//
/*!
 * Initialize Celeritas.
 */
void RunAction::BeginOfRunAction(const G4Run* run)
{
    CELER_EXPECT(run);

    if (G4Threading::IsMasterThread())
    {
        if (!CELERITAS_USE_VECGEOM)
        {
            // For testing purposes, pass the GDML input filename to Celeritas
            const_cast<celeritas::SetupOptions&>(*options_).geometry_file
                = GlobalSetup::Instance()->GetGeometryFile();
        }

        // Create the along-step action
        GlobalSetup::Instance()->SetAlongStep(NoFieldAlongStepFactory{});
    }

    celeritas::ExceptionConverter call_g4exception{"celer0001"};
    CELER_TRY_ELSE(
        {
            // Initialize shared data
            params_->Initialize(*options_);
            CELER_ASSERT(*params_);

            // Construct thread-local transporter
            *transport_ = celeritas::LocalTransporter(*options_, *params_);
            CELER_ENSURE(*transport_);
        },
        call_g4exception);
}

//---------------------------------------------------------------------------//
/*!
 * Finalize Celeritas.
 */
void RunAction::EndOfRunAction(const G4Run*)
{
    CELER_LOG_LOCAL(status) << "Finalizing Celeritas";
    celeritas::ExceptionConverter call_g4exception{"celer0005"};

    // Deallocate Celeritas state data (ensures that objects are deleted on the
    // thread in which they're created, necessary by some geant4
    // thread-local allocators)
    CELER_TRY_ELSE(transport_->Finalize(), call_g4exception);

    // Clear shared data and write if master thread (when running without MT)
    if (G4Threading::IsMasterThread())
    {
        CELER_TRY_ELSE(params_->Finalize(), call_g4exception);
    }
}

//---------------------------------------------------------------------------//
} // namespace demo_geant
