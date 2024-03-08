//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/CsgUnitBuilder.cc
//---------------------------------------------------------------------------//
#include "CsgUnitBuilder.hh"

#include "corecel/io/Logger.hh"
#include "corecel/io/StreamableVariant.hh"
#include "orange/transform/TransformIO.hh"
#include "orange/transform/TransformSimplifier.hh"

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with an empty unit (which doesn't yet have any elements).
 */
CsgUnitBuilder::CsgUnitBuilder(CsgUnit* u, Tolerance<> const& tol)
    : unit_{u}
    , tol_{tol}
    , insert_surface_{&unit_->surfaces, tol}
    , insert_transform_{&unit_->transforms}
{
    CELER_EXPECT(unit_);
    CELER_EXPECT(unit_->empty());

    // Resize because the tree comes prepopulated with true/false
    unit_->metadata.resize(unit_->tree.size());
}

//---------------------------------------------------------------------------//
/*!
 * Access a bounding zone by ID.
 */
BoundingZone const& CsgUnitBuilder::bounds(NodeId nid) const
{
    CELER_EXPECT(nid < unit_->tree.size());

    auto iter = unit_->regions.find(nid);
    CELER_VALIDATE(iter != unit_->regions.end(),
                   << "cannot access bounds for node " << nid.unchecked_get()
                   << ", which is not a region");
    return iter->second.bounds;
}

//---------------------------------------------------------------------------//
/*!
 * Insert transform with simplification and deduplication.
 */
TransformId CsgUnitBuilder::insert_transform(VariantTransform const& vt)
{
    auto simplified = std::visit(TransformSimplifier(tol_), vt);
    return this->insert_transform_(std::move(simplified));
}

//---------------------------------------------------------------------------//
/*!
 * Set a bounding zone and transform for a node.
 */
void CsgUnitBuilder::insert_region(NodeId n,
                                   BoundingZone const& bzone,
                                   TransformId trans_id)
{
    CELER_EXPECT(n < unit_->tree.size());
    CELER_EXPECT(trans_id < unit_->transforms.size());

    auto&& [iter, inserted]
        = unit_->regions.insert({n, CsgUnit::Region{bzone, trans_id}});
    if (CELERITAS_DEBUG && !inserted)
    {
        // The existing bounding zone *SHOULD BE IDENTICAL*.
        // For now this is a rough check...
        CsgUnit::Region const& existing = iter->second;
        CELER_ASSERT(bzone.negated == existing.bounds.negated);
        CELER_ASSERT(static_cast<bool>(bzone.interior)
                     == static_cast<bool>(existing.bounds.interior));
        CELER_ASSERT(static_cast<bool>(bzone.exterior)
                     == static_cast<bool>(existing.bounds.exterior));
        if (trans_id != existing.transform_id)
        {
            // TODO: we should implement transform soft equivalence
            // TODO: transformed shapes that are later defined as volumes (in
            // an RDV or single-item Join function) may result in the same node
            // with two different transforms.
            CELER_LOG(warning)
                << "While re-inserting region for node " << n.get()
                << ": existing transform "
                << StreamableVariant{this->transform(existing.transform_id)}
                << " differs from new transform "
                << StreamableVariant{this->transform(trans_id)};
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Mark a CSG node as a volume of real space.
 *
 * *After* construction is complete, the list of volumes should be checked for
 * duplicate nodes.
 */
LocalVolumeId CsgUnitBuilder::insert_volume(NodeId n)
{
    CELER_EXPECT(n < unit_->tree.size());

    LocalVolumeId result{static_cast<size_type>(unit_->volumes.size())};

    unit_->volumes.push_back(n);
    unit_->fills.resize(unit_->volumes.size());

    CELER_ENSURE(*unit_);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Fill a volume node with a material.
 */
void CsgUnitBuilder::fill_volume(LocalVolumeId v, MaterialId m)
{
    CELER_EXPECT(v < unit_->fills.size());
    CELER_EXPECT(m);

    unit_->fills[v.unchecked_get()] = m;
}

//---------------------------------------------------------------------------//
/*!
 * Fill a volume node with a daughter.
 */
void CsgUnitBuilder::fill_volume(LocalVolumeId v, UniverseId u)
{
    CELER_EXPECT(v < unit_->fills.size());
    CELER_EXPECT(!is_filled(unit_->fills[v.unchecked_get()]));
    CELER_EXPECT(u);

    // Get the region associated with this node ID for the volume
    CELER_ASSERT(unit_->volumes.size() == unit_->fills.size());
    auto iter = unit_->regions.find(unit_->volumes[v.unchecked_get()]);
    // The iterator should be valid since the whole volume should never be a
    // pure surface (created outside a convex region)
    CELER_ASSERT(iter != unit_->regions.end());

    Daughter new_daughter;
    new_daughter.universe_id = u;
    new_daughter.transform_id = iter->second.transform_id;
    CELER_ASSERT(new_daughter.transform_id < unit_->transforms.size());

    // Save fill
    unit_->fills[v.unchecked_get()] = std::move(new_daughter);

    CELER_ENSURE(is_filled(unit_->fills[v.unchecked_get()]));
}

//---------------------------------------------------------------------------//
/*!
 * Get a variant surface from a node ID.
 */
VariantSurface const& CsgUnitBuilder::get_surface_impl(NodeId nid) const
{
    CELER_EXPECT(nid < unit_->tree.size());

    using SurfaceNode = ::celeritas::orangeinp::Surface;

    // Get the surface ID from the tree
    auto const& node = unit_->tree[nid];
    CELER_ASSUME(std::holds_alternative<SurfaceNode>(node));
    LocalSurfaceId lsid = std::get<SurfaceNode>(node).id;

    // Get the variant surfaces from the unit
    CELER_EXPECT(lsid < unit_->surfaces.size());
    return unit_->surfaces[lsid.unchecked_get()];
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
