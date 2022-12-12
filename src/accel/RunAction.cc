//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/RunAction.cc
//---------------------------------------------------------------------------//
#include "RunAction.hh"

#include <CLHEP/Random/Random.h>
#include <G4AutoLock.hh>
#include <G4Run.hh>
#include <G4TransportationManager.hh>

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/ext/GeantImporter.hh"
#include "celeritas/geo/GeoMaterialParams.hh"
#include "celeritas/geo/GeoParams.hh"
#include "celeritas/global/ActionRegistry.hh"
#include "celeritas/global/alongstep/AlongStepGeneralLinearAction.hh"
#include "celeritas/io/ImportProcess.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/phys/CutoffParams.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/PhysicsParams.hh"
#include "celeritas/phys/ProcessBuilder.hh"
#include "celeritas/random/RngParams.hh"
#include "celeritas/track/TrackInitParams.hh"

namespace
{
G4Mutex mutex = G4MUTEX_INITIALIZER;
}

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with Celeritas setup options and shared data.
 */
RunAction::RunAction(SPCOptions options) : options_(options)
{
    CELER_EXPECT(options_);
}

//---------------------------------------------------------------------------//
/*!
 * Initialize Celeritas.
 */
void RunAction::BeginOfRunAction(const G4Run* run)
{
    CELER_EXPECT(run);

    if (false)
    {
        // Maybe the first thread to run: build and store core params
        this->build_core_params();
    }

    // TODO: Construct thread-local transporter
}

//---------------------------------------------------------------------------//
/*!
 * Finalize Celeritas.
 */
void RunAction::EndOfRunAction(const G4Run*) {}

//---------------------------------------------------------------------------//
/*!
 * Finalize Celeritas.
 */
void RunAction::build_core_params()
{
    G4AutoLock lock(&mutex);
    if (false)
    {
        // Some other thread constructed params between the thread-unsafe check
        // and this thread-safe check
        return;
    }
}

//---------------------------------------------------------------------------//
} // namespace celeritas
