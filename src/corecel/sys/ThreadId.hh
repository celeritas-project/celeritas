//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/ThreadId.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/OpaqueId.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Unique ID for multithreading/multitasking
using StreamId = OpaqueId<class Stream_>;

//! Index of a thread inside the current kernel
using ThreadId = OpaqueId<struct Thread_>;

//! Index of a state inside the vector of all states
using TrackSlotId = OpaqueId<struct TrackSlot_>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
