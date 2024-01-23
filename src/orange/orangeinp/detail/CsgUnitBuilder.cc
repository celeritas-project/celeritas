//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/CsgUnitBuilder.cc
//---------------------------------------------------------------------------//
#include "CsgUnitBuilder.hh"

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Set a bounding box for a node.
 */
void CsgUnitBuilder::set_bbox(NodeId n, BBox const& bbox)
{
    CELER_EXPECT(n < unit_->bboxes.size());
    CELER_EXPECT(!unit_->bboxes[n.unchecked_get()]);

    unit_->bboxes[n.unchecked_get()] = bbox;
}

//---------------------------------------------------------------------------//
/*!
 * Mark a CSG node as a volume of real space.
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
void CsgUnitBuilder::fill_volume(LocalVolumeId v,
                                 UniverseId u,
                                 VariantTransform&& vt)
{
    CELER_EXPECT(v < unit_->fills.size());
    CELER_EXPECT(!is_filled(unit_->fills[v.unchecked_get()]));
    CELER_EXPECT(u);

    Daughter new_daughter;
    new_daughter.universe_id = u;
    new_daughter.transform_id
        = TransformId{static_cast<size_type>(unit_->transforms.size())};

    // Add transform
    unit_->transforms.push_back(std::move(vt));
    // Save fill
    unit_->fills[v.unchecked_get()] = std::move(new_daughter);

    CELER_ENSURE(is_filled(unit_->fills[v.unchecked_get()]));
}

//---------------------------------------------------------------------------//
/*!
 * Set an exterior node.
 *
 * This should be called only once (but this could be relaxed if needed).
 */
void CsgUnitBuilder::set_exterior(NodeId n)
{
    CELER_EXPECT(n < unit_->tree.size());
    CELER_EXPECT(!unit_->exterior);

    unit_->exterior = n;
}

//---------------------------------------------------------------------------//
/*!
 * Get a variant surface from a node ID.
 */
VariantSurface const& CsgUnitBuilder::get_surface_impl(NodeId nid) const
{
    CELER_EXPECT(nid < unit_->tree.size());

    using SurfaceNode = ::celeritas::csg::Surface;

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
