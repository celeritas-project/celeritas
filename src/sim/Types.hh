//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Types.hh
//! Type definitions for simulation management
//---------------------------------------------------------------------------//
#pragma once

#include "base/OpaqueId.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Counter for the initiating event for a track
using EventId = OpaqueId<struct Event>;
//! Unique ID (for an event) of a track among all primaries and secondaries
using TrackId = OpaqueId<struct Track>;

//---------------------------------------------------------------------------//
} // namespace celeritas
