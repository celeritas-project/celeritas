//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeInput.hh
//---------------------------------------------------------------------------//
#pragma once

#include <algorithm>
#include <iosfwd>
#include <map>
#include <variant>
#include <vector>

#include "corecel/io/Label.hh"
#include "geocel/BoundingBox.hh"

#include "OrangeData.hh"
#include "OrangeTypes.hh"
#include "surf/VariantSurface.hh"
#include "transform/VariantTransform.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Input definition for a single oriented bounding zone.
 */
struct OrientedBoundingZoneInput
{
    //! Inner bounding box
    BBox inner;
    //! Outer bounding box
    BBox outer;
    //! Local to global transformation
    TransformId trans_id;

    //! Whether the obz definition is valid
    explicit operator bool() const { return inner && outer && trans_id; }
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
    //! OrientedBoundingZone
    OrientedBoundingZoneInput obz;

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
    using MapVolumeDaughter = std::map<LocalVolumeId, DaughterInput>;

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
    explicit operator bool() const { return !universes.empty() && tol; }
};

//---------------------------------------------------------------------------//
// Helper to read the input from a file or stream
std::istream& operator>>(std::istream& is, OrangeInput&);

//---------------------------------------------------------------------------//
// Helper to write the input to a file or stream
std::ostream& operator<<(std::ostream& os, OrangeInput const&);

//---------------------------------------------------------------------------//
}  // namespace celeritas
