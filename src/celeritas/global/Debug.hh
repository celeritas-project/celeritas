//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/Debug.hh
//! \brief Utilities *only* for interactive debugging
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
class CoreParams;
class SimTrackView;

//---------------------------------------------------------------------------//
// Params during an execute call, ONLY for interactive debugging
extern CoreParams const* g_debug_executing_params;

//---------------------------------------------------------------------------//
// Print a SimTrackView on host
void debug_print(SimTrackView const&);

//---------------------------------------------------------------------------//
}  // namespace celeritas
