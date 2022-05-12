//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/launcher/ApplyCutoffLauncher.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "celeritas/phys/Interaction.hh"

namespace celeritas
{
class CoreTrackView;
}

namespace celeritas_test
{
struct ApplyCutoffData;
//---------------------------------------------------------------------------//
/*!
 * Kill the current track.
 */
inline CELER_FUNCTION celeritas::Interaction
                      apply_cutoff_interact_track(ApplyCutoffData const&,
                                                  celeritas::CoreTrackView const&)
{
    return celeritas::Interaction::from_absorption();
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test
