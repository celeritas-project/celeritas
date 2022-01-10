//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VolumeInput.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Span.hh"
#include "../Data.hh"
#include "../Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Input definition for a single volume.
 */
struct VolumeInput
{
    using Flags = VolumeDef::Flags;

    //! Sorted list of surface IDs in this cell
    std::vector<SurfaceId> faces{};
    //! RPN region definition for this cell, using local surface index
    std::vector<logic_int> logic{};

    //! Total number of surface intersections possible in this volume
    logic_int num_intersections{0};
    //! Special flags
    logic_int flags{0};

    //! Whether the volume definition is valid
    explicit operator bool() const { return !logic.empty(); }
};

//---------------------------------------------------------------------------//
} // namespace celeritas
