//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/SimpleOffload.cc
//---------------------------------------------------------------------------//
#include "SimpleOffload.hh"

#include <G4RunManager.hh>

#include "corecel/io/Logger.hh"
#include "corecel/sys/Environment.hh"
#include "corecel/sys/TypeDemangler.hh"

#include "ExceptionConverter.hh"
#include "LocalTransporter.hh"
#include "Logger.hh"
#include "SharedParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from a reference to shared params and local data.
 *
 * On construction, this will check for the \c CELER_DISABLE variable and
 * disable offloading if set. Otherwise it will initialize the multithread
 * logging if the run manager is initialized.
 */
SimpleOffload::SimpleOffload(SetupOptions const* setup,
                             SharedParams* params,
                             LocalTransporter* local)
    : setup_(setup), params_(params), local_(local)
{
    CELER_EXPECT(setup_ && params_);
    CELER_EXPECT((local_ != nullptr)
                 == (G4Threading::IsWorkerThread()
                     || !G4Threading::IsMultithreadedApplication()));

    if (G4Threading::IsMasterThread())
    {
        if (auto* run_man = G4RunManager::GetRunManager())
        {
            // Initialize multithread logger if run manager exists
            celeritas::self_logger() = celeritas::MakeMTLogger(*run_man);
        }
    }
    if (SharedParams::CeleritasDisabled())
    {
        if (G4Threading::IsMasterThread())
        {
            CELER_LOG(info)
                << "Disabling Celeritas offloading since the 'CELER_DISABLE' "
                   "environment variable is present and non-empty";
        }
        else
        {
            CELER_LOG(debug) << "Disabling Celeritas offloading";
        }
        *this = {};
        CELER_ENSURE(!*this);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Initialize celeritas data from setup options.
 */
void SimpleOffload::BeginOfRunAction(G4Run const*)
{
    if (!*this)
        return;

    ExceptionConverter call_g4exception{"celer0001"};

    if (G4Threading::IsMasterThread())
    {
        CELER_TRY_HANDLE(params_->Initialize(*setup_), call_g4exception);
    }
    else
    {
        CELER_TRY_HANDLE(celeritas::SharedParams::InitializeWorker(*setup_),
                         call_g4exception);
    }

    if (local_)
    {
        CELER_LOG_LOCAL(status) << "Constructing local state";
        CELER_TRY_HANDLE(local_->Initialize(*setup_, *params_),
                         call_g4exception);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Send Celeritas the event ID and reseed the Celeritas RNG.
 */
void SimpleOffload::BeginOfEventAction(G4Event const* event)
{
    if (!*this)
        return;

    // Set event ID in local transporter and reseed RNG for reproducibility
    ExceptionConverter call_g4exception{"celer0002"};
    CELER_TRY_HANDLE(local_->InitializeEvent(event->GetEventID()),
                     call_g4exception);
}

//---------------------------------------------------------------------------//
/*!
 * Send tracks to Celeritas if applicable and "StopAndKill" if so.
 */
void SimpleOffload::PreUserTrackingAction(G4Track* track)
{
    if (!*this && !SharedParams::KillOffloadTracks())
        return;

    if (std::find(params_->OffloadParticles().begin(),
                  params_->OffloadParticles().end(),
                  track->GetDefinition())
        != params_->OffloadParticles().end())
    {
        if (!SharedParams::CeleritasDisabled())
        {
            // Celeritas is transporting this track
            ExceptionConverter call_g4exception{"celer0003", params_};
            CELER_TRY_HANDLE(local_->Push(*track), call_g4exception);
        }
        track->SetTrackStatus(fStopAndKill);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Flush offloaded tracks from Celeritas.
 */
void SimpleOffload::EndOfEventAction(G4Event const*)
{
    if (!*this)
        return;

    ExceptionConverter call_g4exception{"celer0004", params_};
    CELER_TRY_HANDLE(local_->Flush(), call_g4exception);
}

//---------------------------------------------------------------------------//
/*!
 * Finalize Celeritas.
 */
void SimpleOffload::EndOfRunAction(G4Run const*)
{
    if (!*this)
        return;

    CELER_LOG_LOCAL(status) << "Finalizing Celeritas";
    ExceptionConverter call_g4exception{"celer0005"};

    if (local_)
    {
        CELER_TRY_HANDLE(local_->Finalize(), call_g4exception);
    }

    if (G4Threading::IsMasterThread())
    {
        CELER_TRY_HANDLE(params_->Finalize(), call_g4exception);
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
