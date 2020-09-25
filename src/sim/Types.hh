//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Types.hh
//---------------------------------------------------------------------------//
#ifndef sim_Types_hh
#define sim_Types_hh

#include "base/OpaqueId.hh"

namespace celeritas
{
struct Event;
struct Track;
//---------------------------------------------------------------------------//

using EventId = OpaqueId<Event>;
using TrackId = OpaqueId<Track>;

//---------------------------------------------------------------------------//
} // namespace celeritas

#endif // sim_Types_hh
