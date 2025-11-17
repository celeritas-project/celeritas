//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeInput.hh
//! \todo Move to inp/Orange.hh ?
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
 * Input definition for a single ORANGE implementation volume.
 */
struct VolumeInput
{
    using Flags = VolumeRecord::Flags;
    using VariantLabel = std::variant<Label, VolumeInstanceId>;

    //! Volume label or instance ID
    VariantLabel label{};

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

    //! Whether the input definition is valid
    explicit operator bool() const
    {
        return (!logic.empty() || (flags & Flags::implicit_vol))
               && zorder != ZOrder::invalid;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Input definition a daughter universe embedded in a parent volume.
 */
struct DaughterInput
{
    UnivId univ_id;
    VariantTransform transform;
};

//---------------------------------------------------------------------------//
/*!
 * Extra metadata for a unit's "background" volume.
 *
 * Unlike a regular volume, the "background" represents a \em volume rather
 * than a volume \em instance. Note that this can be an \em explicit volume
 * (i.e., made of booleans) or \em implicit (i.e., have the lowest "Z order").
 *
 * This is something of a hack: the background volume in a \c
 * orangeinp::UnitProto is annotated by setting the label to \c
 * VolumeInstanceId{} in \c g4org::ProtoConstructor; then converted from a
 * proto to a \c UnitInput by the \c InputBuilder, and finally in \c
 * g4org::Converter the empty volume instance IDs are replaced by (1) the
 * world \c VolumeInstanceId for the top-level background volume, or (2) the
 * \c VolumeId corresponding to the unit's label.
 */
struct BackgroundInput
{
    VolumeId label;
    LocalVolumeId volume;

    //! Whether the background metadata is used
    explicit operator bool() const { return static_cast<bool>(volume); }
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
    using MapLocalParent = std::map<LocalVolumeId, LocalVolumeId>;

    std::vector<VariantSurface> surfaces;
    std::vector<VolumeInput> volumes;
    //! Outer bounding box
    BBox bbox;

    //! The given local volume is replaced by a transformed universe
    MapVolumeDaughter daughter_map;
    //! The given local volume is structurally "inside" another local volume
    MapLocalParent local_parent_map;
    //! Metadata for the volume that represents the boundary of the unit
    BackgroundInput background;

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

    // Daughters in each volume [x][y][z]
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

    //! Logic expression notation
    LogicNotation logic{LogicNotation::postfix};

    //! Whether the unit definition is valid
    explicit operator bool() const
    {
        return !universes.empty() && tol && logic != LogicNotation::size_;
    }
};

//---------------------------------------------------------------------------//
// Helper to read the input from a file or stream
std::istream& operator>>(std::istream& is, OrangeInput&);

//---------------------------------------------------------------------------//
// Helper to write the input to a file or stream
std::ostream& operator<<(std::ostream& os, OrangeInput const&);

//---------------------------------------------------------------------------//
}  // namespace celeritas
