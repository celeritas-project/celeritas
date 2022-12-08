//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/RunAction.cc
//---------------------------------------------------------------------------//
#include "RunAction.hh"

#include <G4Run.hh>

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"
#include <CLHEP/Random/Random.h>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with Celeritas setup options.
 */
RunAction::RunAction(SPCOptions options) : options_(std::move(options))
{
    CELER_EXPECT(options_);
    CELER_LOG_LOCAL(debug) << "RunAction::RunAction";
}

//---------------------------------------------------------------------------//
/*!
 * Initialize Celeritas.
 */
void RunAction::BeginOfRunAction(const G4Run* run)
{
    CELER_EXPECT(run);
    CELER_LOG_LOCAL(debug) << "RunAction::BeginOfRunAction for run "
                           << run->GetRunID()
                           << (this->IsMaster() ? " (master)" : "");

    // TODO: set RNG seed via CLHEP::HepRandom::getTheSeed();
    // TODO: set up physics and geometry if master?
    // or if first thread to hit (via mutex?)
}

//---------------------------------------------------------------------------//
/*!
 * Finalize Celeritas.
 */
void RunAction::EndOfRunAction(const G4Run*)
{
    CELER_LOG_LOCAL(debug) << "RunAction::EndOfRunAction";
}

//---------------------------------------------------------------------------//
} // namespace celeritas
