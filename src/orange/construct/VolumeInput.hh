//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VolumeInput.hh
//---------------------------------------------------------------------------//
#pragma once

#include "../Data.hh"
#include "../Types.hh"
#include "base/Span.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Input definition for a single volume.
 */
struct VolumeInput
{
    using Flags = VolumeRecord::Flags;

    //! Sorted list of surface IDs in this cell
    std::vector<SurfaceId> faces{};
    //! RPN region definition for this cell, using local surface index
    std::vector<logic_int> logic{};

    //! Maximum possible number of surface intersections in this volume
    logic_int max_intersections{0};
    //! Special flags
    logic_int flags{0};

    //! Whether the volume definition is valid
    explicit operator bool() const { return !logic.empty(); }
};

//---------------------------------------------------------------------------//
} // namespace celeritas
