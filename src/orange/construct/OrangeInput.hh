//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
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
#include "orange/surf/VariantSurface.hh"
#include "orange/transform/VariantTransform.hh"

namespace celeritas
{
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
    //! Masking priority
    ZOrder zorder{};

    //! Whether the volume definition is valid
    explicit operator bool() const
    {
        return (!logic.empty() || (flags & Flags::implicit_vol))
               && zorder != ZOrder::invalid;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Input definition a daughter universe embedded in a parent cell.
 */
struct DaughterInput
{
    UniverseId universe_id;
    VariantTransform transform;
};

//---------------------------------------------------------------------------//
/*!
 * Input definition for a unit.
 *
 * \todo Add a CsgTree object and \c vector<NodeId> volumes;
 */
struct UnitInput
{
    using MapVolumeDaughter = std::unordered_map<LocalVolumeId, DaughterInput>;

    std::vector<VariantSurface> surfaces;
    std::vector<VolumeInput> volumes;
    BBox bbox;  //!< Outer bounding box
    MapVolumeDaughter daughter_map;

    // Unit metadata
    std::vector<Label> surface_labels;
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
//! Possible types of universe inputs
using VariantUniverseInput = std::variant<UnitInput, RectArrayInput>;

//---------------------------------------------------------------------------//
/*!
 * Construction definition for a full ORANGE geometry.
 */
struct OrangeInput
{
    std::vector<VariantUniverseInput> universes;

    //! Relative and absolute error for construction and transport
    Tolerance<> tol;

    //! Whether the unit definition is valid
    explicit operator bool() const { return !universes.empty(); }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
