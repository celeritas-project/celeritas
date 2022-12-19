//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/RunAction.cc
//---------------------------------------------------------------------------//
#include "RunAction.hh"

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

    // Initialize shared data
    params_->Initialize(*options_);
    CELER_ASSERT(*params_);

    // Construct thread-local transporter
    *transport_ = celeritas::LocalTransporter(*options_, *params_);
    CELER_ENSURE(*transport_);
}

//---------------------------------------------------------------------------//
/*!
 * Finalize Celeritas.
 */
void RunAction::EndOfRunAction(const G4Run*)
{
    // Deallocate Celeritas state data (optional)
    transport_.reset();
}

//---------------------------------------------------------------------------//
} // namespace demo_geant
