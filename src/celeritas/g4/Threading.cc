//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/g4/Threading.cc
//---------------------------------------------------------------------------//
#include "Threading.hh"

#include <G4Threading.hh>

#include "geocel/GeantUtils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Get the worker stream ID for the program's \em main thread (null if MT).
 *
 * This can be compared against \c geant_stream . In serial execution, the
 * main stream is also a "worker" thread (running events). In MT/tasking,
 * the main stream is a separate \em manager thread, which does \em not
 * run events.
 */
StreamId geant_main_stream()
{
    if (!G4Threading::IsMultithreadedApplication())
    {
        return StreamId{0};
    }
    return StreamId{};
}

//---------------------------------------------------------------------------//
/*!
 * Get a stream ID corresponding to the current worker thread.
 *
 * The result is null if this is the "master" thread in MT or if the run
 * manager hasn't been started.
 */
StreamId geant_stream()
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
/*!
 * Get the number of threads after initialization.
 */
StreamId::size_type geant_num_threads()
{
    int result = get_geant_num_threads();
    CELER_ENSURE(result > 0);
    return static_cast<StreamId::size_type>(result);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
