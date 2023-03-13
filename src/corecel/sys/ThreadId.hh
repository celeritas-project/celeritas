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
// Forward declare meaningless struct to avoid conflict with globally
// namespaced PTL::Thread class when defining ThreadId type.
struct Thread;
struct TrackSlot;

//---------------------------------------------------------------------------//
//! Index of a thread inside the current kernel
using ThreadId = OpaqueId<struct Thread>;

//! Index of a state inside the vector of all states
using TrackSlotId = OpaqueId<struct TrackSlot>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
