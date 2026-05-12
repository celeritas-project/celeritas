//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/g4/Threading.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/sys/ThreadId.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//

// Get the worker stream ID for the program's main thread (null if MT)
StreamId geant_main_stream();

// Get a stream ID corresponding to the current Geant4 worker thread
StreamId geant_stream();

// Get the number of threads after initialization
StreamId::size_type geant_num_threads();

//---------------------------------------------------------------------------//
}  // namespace celeritas
