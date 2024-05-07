//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/Utils.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/global/CoreTrackView.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// Get the optical material ID, or an invalid ID if the track is inactive
inline CELER_FUNCTION OpticalMaterialId
get_optical_material(CoreTrackView const& track)
{
    if (track.make_sim_view().status() == TrackStatus::inactive)
        return {};

    return track.make_material_view().make_material_view().optical_material_id();
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
