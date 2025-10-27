//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/CsgUnitBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/io/Label.hh"
#include "orange/OrangeTypes.hh"

#include "CsgUnit.hh"
#include "LocalSurfaceInserter.hh"
#include "TransformInserter.hh"
#include "../CsgTypes.hh"

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
 * This *LOW-LEVEL* class keeps track of the CSG and surface nodes during
 * construction. It holds the "construction-time" properties like the local
 * surface inserter. The input "CSG Unit" must exceed the lifetime of this
 * builder.
 *
 * The geometry construction tolerance is passed to numerous other classes
 * during construction. The input bounding box is used to restrict the maximum
 * possible bounding box for any of the child objects.
 *
 * This class is meant to be used by:
 * - Object builders
 * - Convex surface builder
 */
class CsgUnitBuilder
{
  public:
    //!@{
    //! \name Type aliases
    using Tol = Tolerance<>;
    using Metadata = CsgUnit::Metadata;
    using NodeInsertion = CsgTree::Insertion;
    //!@}

  public:
    // Construct with an empty unit, tolerance settings, and a priori bbox
    CsgUnitBuilder(CsgUnit*, Tol const& tol, BBox const& extents);

    //// ACCESSORS ////

    //! Tolerance, needed for surface simplifier
    Tol const& tol() const { return tol_; }

    //! Maximum extents of this unit
    BBox const& extents() const { return bbox_; }

    // Access a typed surface, needed for clipping with deduplicated surface
    template<class S>
    inline S const& surface(NodeId) const;

    // Access a typed CSG node after insertion
    template<class T>
    inline T const& node(NodeId) const;

    // Access a bounding zone by ID
    BoundingZone const& bounds(NodeId) const;

    // Access a transform by ID
    inline VariantTransform const& transform(TransformId) const;

    //// MUTATORS ////

    // Insert a surface by forwarding to the surface inserter
    template<class... Args>
    inline NodeInsertion insert_surface(Args&&... args);

    // Insert a CSG node by forwarding to the CsgTree
    template<class... Args>
    inline NodeInsertion insert_csg(Args&&... args);

    // Insert a transform
    TransformId insert_transform(VariantTransform const& vt);

    // Insert node metadata
    inline void insert_md(NodeId node, Metadata&& md);

    // Set a bounding zone and transform for a node
    void insert_region(NodeId, BoundingZone const&, TransformId trans_id);

    // Mark a CSG node as a volume of real space
    LocalVolumeId insert_volume(NodeId);

    // Fill a volume node with a material
    void fill_volume(LocalVolumeId, GeoMatId);

    // Fill a volume node with a daughter using the local transform
    void fill_volume(LocalVolumeId, UnivId, VariantTransform const& transform);

  private:
    CsgUnit* unit_;
    Tol tol_;
    BBox bbox_;
    LocalSurfaceInserter insert_surface_;
    TransformInserter insert_transform_;

    // Get a variant surface from a node ID
    VariantSurface const& get_surface_impl(NodeId nid) const;

    // TODO: cache of weak_ptr<{Transform,ObjectInterface}> -> NodeId?
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Access a typed surface, needed for clipping with deduplicated surface.
 */
template<class S>
S const& CsgUnitBuilder::surface(NodeId nid) const
{
    VariantSurface const& vs = this->get_surface_impl(nid);
    CELER_ASSUME(std::holds_alternative<S>(vs));
    return std::get<S>(vs);
}

//---------------------------------------------------------------------------//
/*!
 * Access a CSG node after insertion.
 */
template<class T>
T const& CsgUnitBuilder::node(NodeId nid) const
{
    auto const& node = unit_->tree[nid];
    CELER_ASSUME(std::holds_alternative<T>(node));
    return std::get<T>(node);
}

//---------------------------------------------------------------------------//
/*!
 * Access a transform by ID.
 */
VariantTransform const& CsgUnitBuilder::transform(TransformId tid) const
{
    CELER_EXPECT(tid < unit_->transforms.size());
    return unit_->transforms[tid.unchecked_get()];
}

//---------------------------------------------------------------------------//
/*!
 * Insert a surface by forwarding to the surface inserter.
 */
template<class... Args>
auto CsgUnitBuilder::insert_surface(Args&&... args) -> NodeInsertion
{
    LocalSurfaceId lsid = insert_surface_(std::forward<Args>(args)...);
    return this->insert_csg(lsid);
}

//---------------------------------------------------------------------------//
/*!
 * Insert a CSG node by forwarding to the CsgTree.
 */
template<class... Args>
auto CsgUnitBuilder::insert_csg(Args&&... args) -> NodeInsertion
{
    auto result = unit_->tree.insert(std::forward<Args>(args)...);
    if (result.second)
    {
        unit_->metadata.resize(unit_->tree.size());
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
