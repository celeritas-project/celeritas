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

// Get a stream ID corresponding to the current worker thread
StreamId g4_worker_stream();

//---------------------------------------------------------------------------//
}  // namespace celeritas
