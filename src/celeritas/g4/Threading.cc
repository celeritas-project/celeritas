//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/g4/Threading.cc
//---------------------------------------------------------------------------//
#include "Threading.hh"

#include <G4Threading.hh>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Get a stream ID corresponding to the current worker thread.
 *
 * The result is null if this is the "master" thread in MT or if the run
 * manager hasn't been started.
 */
StreamId g4_worker_stream()
{
    if (!G4Threading::IsMultithreadedApplication())
    {
        return StreamId{0};
    }
    if (G4Threading::IsMasterThread())
    {
        return {};
    }
    int tid = G4Threading::G4GetThreadId();
    CELER_ASSERT(tid >= 0);
    return id_cast<StreamId>(tid);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
