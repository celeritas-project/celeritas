//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file example/accel/offload-template/src/RunAction.cc
//---------------------------------------------------------------------------//
#include "RunAction.hh"

#include <accel/ExceptionConverter.hh>

#include "Celeritas.hh"
#include "G4Threading.hh"

//---------------------------------------------------------------------------//
/*!
 * Construct empty.
 */
RunAction::RunAction() : G4UserRunAction() {}

//---------------------------------------------------------------------------//
/*!
 * Initialize master and worker threads in Celeritas.
 *
 * \note
 * This minimal template is single-threaded and thus the MT case is here just
 * for reference.
 */
void RunAction::BeginOfRunAction(G4Run const*)
{
    celeritas::ExceptionConverter HandleExceptions{"celer0001"};
    if (G4Threading::IsMasterThread())
    {
        CELER_TRY_HANDLE(CelerSharedParams().Initialize(CelerSetupOptions()),
                         HandleExceptions);
    }
    else
    {
        CELER_TRY_HANDLE(
            celeritas::SharedParams::InitializeWorker(CelerSetupOptions()),
            HandleExceptions);
    }

    if (G4Threading::IsWorkerThread()
        || !G4Threading::IsMultithreadedApplication())
    {
        CELER_TRY_HANDLE(CelerLocalTransporter().Initialize(
                             CelerSetupOptions(), CelerSharedParams()),
                         HandleExceptions);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Clear local data and return Celeritas to an invalid state.
 */
void RunAction::EndOfRunAction(G4Run const*)
{
    celeritas::ExceptionConverter HandleExceptions{"celer0005"};

    if (CelerLocalTransporter())
    {
        CELER_TRY_HANDLE(CelerLocalTransporter().Finalize(), HandleExceptions);
    }

    if (G4Threading::IsMasterThread())
    {
        CELER_TRY_HANDLE(CelerSharedParams().Finalize(), HandleExceptions);
    }
}
