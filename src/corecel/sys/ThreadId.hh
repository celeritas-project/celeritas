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
// Forward declare types to avoid potential conflicts (e.g. PTL::Thread)
class Stream;
struct Thread;
struct TrackSlot;

//---------------------------------------------------------------------------//
//! Unique ID for multithreading/multitasking
using StreamId = OpaqueId<class Stream>;

//! Index of a thread inside the current kernel
using ThreadId = OpaqueId<struct Thread>;

//! Index of a state inside the vector of all states
using TrackSlotId = OpaqueId<struct TrackSlot>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
