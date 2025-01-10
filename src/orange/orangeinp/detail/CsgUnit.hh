//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/CsgUnit.hh
//---------------------------------------------------------------------------//
#pragma once

#include <map>
#include <set>
#include <variant>
#include <vector>

#include "corecel/io/Label.hh"
#include "geocel/BoundingBox.hh"
#include "orange/OrangeTypes.hh"
#include "orange/surf/VariantSurface.hh"
#include "orange/transform/VariantTransform.hh"

#include "BoundingZone.hh"
#include "../CsgTree.hh"
#include "../CsgTypes.hh"

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Constructed CSG geometry data for a unit.
 *
 * This contains all the construction-time surfaces, volumes, and properties.
 * These are stored in a way so that they can be remapped and/or optimized
 * further before committing them to the constructed GPU data.
 *
 * All bounding boxes and transforms are "local" within the CSG unit's
 * reference frame, not relative to any other CSG node nor to any parent
 * universe. (TODO: add bounds and transforms only for finite regions)
 *
 * TODO (?) map nodes to set of object pointers, for detailed provenance?
 */
struct CsgUnit
{
    //// TYPES ////

    using Metadata = Label;
    using SetMd = std::set<Metadata>;
    using Fill = std::variant<std::monostate, GeoMaterialId, Daughter>;

    //! Attributes about a closed volume of space
    struct Region
    {
        BoundingZone bounds;  //!< Interior/exterior bbox
        TransformId trans_id;  //!< Region-to-unit transform
    };

    //// DATA ////

    //!@{
    //! \name Surfaces
    //! Vectors are indexed by LocalSurfaceId.
    std::vector<VariantSurface> surfaces;
    //!@}

    //!@{
    //! \name Nodes
    //! Vectors are indexed by NodeId.
    CsgTree tree;  //!< CSG tree
    std::vector<SetMd> metadata;  //!< CSG node labels
    std::map<NodeId, Region> regions;  //!< Bounds and transforms
    //!@}

    //!@{
    //! \name Volumes
    //! Vector is indexed by LocalVolumeId.
    std::vector<Fill> fills;  //!< Content of each volume
    GeoMaterialId background;  //!< Optional background fill
    //!@}

    //!@{
    //! \name Transforms
    //! Vectors are indexed by TransformId.
    std::vector<VariantTransform> transforms;
    //!@}

    //// FUNCTIONS ////

    // Whether the processed unit is valid for use
    explicit inline operator bool() const;

    // Whether the unit has no constructed data
    inline bool empty() const;
};

//---------------------------------------------------------------------------//
/*!
 * Utility for telling whether a fill is assigned.
 */
inline constexpr bool is_filled(CsgUnit::Fill const& fill)
{
    return !std::holds_alternative<std::monostate>(fill);
}

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Whether the processed unit is valid for use.
 */
CsgUnit::operator bool() const
{
    return this->metadata.size() == this->tree.size()
           && !this->tree.volumes().empty()
           && this->tree.volumes().size() == this->fills.size();
    ;
}

//---------------------------------------------------------------------------//
/*!
 * True if the unit has no constructed data.
 */
bool CsgUnit::empty() const
{
    return this->surfaces.empty() && this->metadata.empty()
           && this->regions.empty() && this->tree.volumes().empty()
           && this->fills.empty() && this->transforms.empty();
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
