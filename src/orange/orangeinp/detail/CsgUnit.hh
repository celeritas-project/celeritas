//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/CsgUnit.hh
//---------------------------------------------------------------------------//
#pragma once

#include <set>
#include <variant>
#include <vector>

#include "corecel/io/Label.hh"
#include "orange/BoundingBox.hh"
#include "orange/OrangeTypes.hh"
#include "orange/construct/CsgTree.hh"
#include "orange/construct/CsgTypes.hh"
#include "orange/surf/VariantSurface.hh"

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
 * The "exterior" is optional *only* in the degenerate case of an infinite
 * global universe (TODO: prohibit this??)
 *
 * TODO: improve metadata (provenance, nicer container, mapping?, calculated
 * volumes)
 */
struct CsgUnit
{
    //// TYPES ////

    using Metadata = Label;
    using SetMd = std::set<Metadata>;
    using NodeId = ::celeritas::csg::NodeId;
    using BBox = ::celeritas::BoundingBox<>;
    using Fill = std::variant<std::monostate, MaterialId, Daughter>;

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
    std::vector<SetMd> metadata;  //!< CSG node labels and provenance
    std::vector<BBox> bboxes;
    //!@}

    //!@{
    //! \name Volumes
    //! Vectors are indexed by LocalVolumeId.
    std::vector<NodeId> volumes;  //!< CSG node of each volume
    std::vector<Fill> fills;  //!< Content of each volume
    NodeId exterior;
    //!@}

    //!@{
    //! \name Transforms
    //! Vectors are indexed by TransformId.
    std::vector<VariantTransform> transforms;
    //!@}

    //// FUNCTIONS ////

    // Whether the processed unit is valid for use
    explicit inline operator bool() const;
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
           && this->bboxes.size() == this->tree.size() && !this->volumes.empty()
           && this->volumes.size() == this->fills.size();
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
