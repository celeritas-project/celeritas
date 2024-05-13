//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/CsgUnitBuilder.cc
//---------------------------------------------------------------------------//
#include "CsgUnitBuilder.hh"

#include "corecel/io/Join.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/StreamableVariant.hh"
#include "orange/OrangeData.hh"
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
 * Construct with an empty unit, tolerance settings, and a priori extents.
 *
 * The unit should have no elements to start with.
 */
CsgUnitBuilder::CsgUnitBuilder(CsgUnit* u,
                               Tolerance<> const& tol,
                               BBox const& extents)
    : unit_{u}
    , tol_{tol}
    , bbox_{extents}
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
    if (!inserted)
    {
        // The existing bounding zone *SHOULD BE IDENTICAL* since it's the same
        // CSG definition
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
            // with two different transforms. These transforms don't usually
            // matter though?
            auto const& md = unit_->metadata[n.get()];
            CELER_LOG(debug)
                << "While re-inserting logically equivalent region '"
                << join(md.begin(), md.end(), "' = '")
                << "': existing transform "
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
 * Fill LocalVolumeId{0} with "exterior" to adjust the interior region.
 *
 * This should be called to process the exterior volume *immediately* after its
 * creation.
 */
void CsgUnitBuilder::fill_exterior()
{
    CELER_EXPECT(unit_->volumes.size() == 1);
    static_assert(orange_exterior_volume == LocalVolumeId{0});

    NodeId n = unit_->volumes[orange_exterior_volume.get()];
    auto iter = unit_->regions.find(n);
    CELER_ASSERT(iter != unit_->regions.end());
    CELER_VALIDATE(!iter->second.bounds.negated,
                   << "exterior volume is inside out");

    // TODO handle edge case where exterior is the composite of two volumes and
    // we need to adjust those volumes' bboxes?
    bbox_ = calc_intersection(bbox_, iter->second.bounds.exterior);
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
 *
 * The transform is from the current universe to the daughter. The
 * corresponding shape may have additional transforms as well.
 */
void CsgUnitBuilder::fill_volume(LocalVolumeId v,
                                 UniverseId u,
                                 VariantTransform const& transform)
{
    CELER_EXPECT(v < unit_->fills.size());
    CELER_EXPECT(!is_filled(unit_->fills[v.unchecked_get()]));
    CELER_EXPECT(u);

    Daughter new_daughter;
    new_daughter.universe_id = u;
    new_daughter.transform_id = this->insert_transform(transform);
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
