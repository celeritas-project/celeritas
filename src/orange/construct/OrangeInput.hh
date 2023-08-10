//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/construct/OrangeInput.hh
//---------------------------------------------------------------------------//
#pragma once

#include <algorithm>
#include <unordered_map>  // IWYU pragma: export
#include <variant>
#include <vector>

#include "corecel/io/Label.hh"
#include "orange/BoundingBox.hh"
#include "orange/OrangeData.hh"
#include "orange/OrangeTypes.hh"
#include "orange/Translator.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Compressed input for all surface definitions in a unit.
 *
 * Including the sizes of each surface is redundant but safer.
 */
struct SurfaceInput
{
    std::vector<SurfaceType> types;  //!< Surface type enums
    std::vector<real_type> data;  //!< Compressed surface data
    std::vector<size_type> sizes;  //!< Size of each surface's data
    std::vector<Label> labels;  //!< Surface labels

    //! Number of surfaces
    size_type size() const { return types.size(); }

    //! Whether the surface inputs are valid
    explicit operator bool() const
    {
        return types.size() == sizes.size() && labels.size() == types.size()
               && data.size() >= types.size();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Input definition for a single volume.
 */
struct VolumeInput
{
    using Flags = VolumeRecord::Flags;

    //! Volume label
    Label label{};

    //! Sorted list of surface IDs in this volume
    std::vector<LocalSurfaceId> faces{};
    //! RPN region definition for this volume, using local surface index
    std::vector<logic_int> logic{};
    //! Axis-aligned bounding box
    BBox bbox{};

    //! Special flags
    logic_int flags{0};
    //! Masking priority (2 for regular, 1 for background)
    int zorder{};

    //! Whether the volume definition is valid
    explicit operator bool() const
    {
        return !logic.empty() || (flags & Flags::implicit_vol);
    }
};

//---------------------------------------------------------------------------//
/*!
 * Input definition a daughter universe embedded in a parent cell
 */
struct DaughterInput
{
    UniverseId universe_id;
    Translation translation;
};

//---------------------------------------------------------------------------//
/*!
 * Input definition for a unit.
 */
struct UnitInput
{
    using MapVolumeDaughter = std::unordered_map<LocalVolumeId, DaughterInput>;

    SurfaceInput surfaces;
    std::vector<VolumeInput> volumes;
    BBox bbox;  //!< Outer bounding box
    MapVolumeDaughter daughter_map;

    // Unit metadata
    Label label;

    //! Whether the unit definition is valid
    explicit operator bool() const { return !volumes.empty(); }
};

//---------------------------------------------------------------------------//
/*!
 * Input definition for a rectangular array universe.
 */
struct RectArrayInput
{
    // Grid boundaries in x, y, and z
    Array<std::vector<double>, 3> grid;

    // Daughters in each cell [x][y][z]
    std::vector<DaughterInput> daughters;

    // Unit metadata
    Label label;

    //! Whether the universe definition is valid
    explicit operator bool() const
    {
        return !daughters.empty()
               && std::all_of(grid.begin(), grid.end(), [](auto& v) {
                      return v.size() >= 2;
                  });
    }
};

//---------------------------------------------------------------------------//
/*!
 * Construction definition for a full ORANGE geometry.
 */
struct OrangeInput
{
    std::vector<std::variant<UnitInput, RectArrayInput>> universes;

    // TODO: Calculate automatically in Shift by traversing the parent/daughter
    // tree
    size_type max_level = 3;

    // TODO: array of universe types and universe ID -> offset
    // or maybe std::variant when we require C++17

    //! Whether the unit definition is valid
    explicit operator bool() const
    {
        return !universes.empty() && max_level > 0;
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
