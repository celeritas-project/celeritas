//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/CsgUnitBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/io/Label.hh"
#include "orange/OrangeTypes.hh"
#include "orange/construct/CsgTypes.hh"
#include "orange/construct/detail/LocalSurfaceInserter.hh"

#include "CsgUnit.hh"

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Help construct a unit's mutable CSG representation.
 *
 * This keeps track of the CSG and surface nodes during construction. It holds
 * the "construction-time" properties like the local surface inserter. The
 * input "CSG Unit" must exceed the lifetime of this builder.
 */
class CsgUnitBuilder
{
  public:
    //!@{
    //! \name Type aliases
    using Tol = Tolerance<>;
    using Metadata = CsgUnit::Metadata;
    using NodeId = CsgUnit::NodeId;
    using BBox = CsgUnit::BBox;
    //!@}

  public:
    // Construct with an empty unit and tolerance settings
    inline CsgUnitBuilder(CsgUnit*, Tol const& tol);

    //// ACCESSORS ////

    //! Tolerance, needed for surface simplifier
    Tol const& tol() const { return tol_; }

    // Access a typed surface, needed for clipping with deduplicated surface
    template<class S>
    inline S const& get_surface(NodeId) const;

    //// MUTATORS ////

    // Insert a surface by forwarding to the surface inserter
    template<class... Args>
    inline NodeId insert_surface(Args&&... args);

    // Insert a CSG node by forwarding to the CsgTree
    template<class... Args>
    inline NodeId insert_csg(Args&&... args);

    //! Insert node metadata
    inline void insert_md(NodeId node, Metadata&& md);

    // Set a bounding box for a node
    void set_bbox(NodeId, BBox const&);

    // Mark a CSG node as a volume of real space
    LocalVolumeId insert_volume(NodeId);

    // Fill a volume node with a material
    void fill_volume(LocalVolumeId, MaterialId);

    // Fill a volume node with a daughter
    void fill_volume(LocalVolumeId, UniverseId, VariantTransform&& vt);

    // Set an exterior node
    void set_exterior(NodeId);

  private:
    using LocalSurfaceInserter = ::celeritas::detail::LocalSurfaceInserter;

    CsgUnit* unit_;
    Tol tol_;
    LocalSurfaceInserter insert_surface_;

    // Get a variant surface from a node ID
    VariantSurface const& get_surface_impl(NodeId nid) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with an empty unit (which doesn't yet have any elements).
 */
CsgUnitBuilder::CsgUnitBuilder(CsgUnit* u, Tolerance<> const& tol)
    : unit_{u}, tol_{tol}, insert_surface_{&unit_->surfaces, tol}
{
    CELER_EXPECT(unit_);
    CELER_EXPECT(unit_->empty());
}

//---------------------------------------------------------------------------//
/*!
 * Access a typed surface, needed for clipping with deduplicated surface.
 */
template<class S>
S const& CsgUnitBuilder::get_surface(NodeId nid) const
{
    VariantSurface const& vs = this->get_surface_impl(nid);
    CELER_ASSUME(std::holds_alternative<S>(vs));
    return std::get<S>(vs);
}

//---------------------------------------------------------------------------//
/*!
 * Insert a surface by forwarding to the surface inserter.
 */
template<class... Args>
auto CsgUnitBuilder::insert_surface(Args&&... args) -> NodeId
{
    LocalSurfaceId lsid = insert_surface_(std::forward<Args>(args)...);
    return this->insert_csg(lsid);
}

//---------------------------------------------------------------------------//
/*!
 * Insert a CSG node by forwarding to the CsgTree.
 */
template<class... Args>
auto CsgUnitBuilder::insert_csg(Args&&... args) -> NodeId
{
    NodeId result = unit_->tree.insert(std::forward<Args>(args)...);
    if (!(result < unit_->metadata.size()))
    {
        unit_->metadata.resize(unit_->tree.size());
        unit_->bboxes.resize(unit_->tree.size());
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Insert node metadata.
 */
void CsgUnitBuilder::insert_md(NodeId node, Metadata&& md)
{
    CELER_EXPECT(node < unit_->metadata.size());
    unit_->metadata[node.unchecked_get()].insert(std::move(md));
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
