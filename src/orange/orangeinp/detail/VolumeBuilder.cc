//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/VolumeBuilder.cc
//---------------------------------------------------------------------------//
#include "VolumeBuilder.hh"

#include "corecel/Assert.hh"

#include "BoundingZone.hh"
#include "CsgUnitBuilder.hh"

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with unit builder.
 */
VolumeBuilder::VolumeBuilder(CsgUnitBuilder* ub) : ub_{ub}
{
    CELER_EXPECT(ub_);
    this->push_transform(NoTransformation{});
    CELER_ENSURE(transforms_.size() == 1);
}

//---------------------------------------------------------------------------//
/*!
 * Get the construction tolerance.
 */
auto VolumeBuilder::tol() const -> Tol const&
{
    return ub_->tol();
}

//---------------------------------------------------------------------------//
/*!
 * Access the local-to-global transform during construction.
 */
VariantTransform const& VolumeBuilder::local_transform() const
{
    return ub_->transform(transforms_.back());
}

//-------------------------------------------------------------------------//
/*!
 * Add a region to the CSG tree, automatically calculating bounding zone.
 */
NodeId VolumeBuilder::insert_region(Metadata&& md, Joined&& j)
{
    // Calculate bounding zone before creating region
    BoundingZone bz = [&j, this] {
        if (j.op == op_and)
        {
            // Shrink an infinite bounding zone
            BoundingZone result = BoundingZone::from_infinite();
            for (NodeId d : j.nodes)
            {
                result = calc_intersection(result, ub_->bounds(d));
            }
            return result;
        }
        else if (j.op == op_or)
        {
            // Grow a null bounding zone
            BoundingZone result;
            for (NodeId d : j.nodes)
            {
                result = calc_union(result, ub_->bounds(d));
            }
            return result;
        }
        else
        {
            CELER_ASSERT_UNREACHABLE();
        }
    }();

    auto node_id = ub_->insert_csg(std::move(j)).first;
    ub_->insert_md(node_id, std::move(md));
    ub_->insert_region(node_id, std::move(bz), transforms_.back());

    return node_id;
}

//-------------------------------------------------------------------------//
/*!
 * Add a region to the CSG tree using a custom bounding zone.
 */
NodeId
VolumeBuilder::insert_region(Metadata&& md, Joined&& j, BoundingZone&& bz)
{
    auto node_id = ub_->insert_csg(std::move(j)).first;
    ub_->insert_md(node_id, std::move(md));
    ub_->insert_region(node_id, std::move(bz), transforms_.back());

    return node_id;
}

//---------------------------------------------------------------------------//
/*!
 * Add a negated region to the CSG tree.
 */
NodeId VolumeBuilder::insert_region(Metadata&& md, Negated&& n)
{
    CELER_EXPECT(n.node);

    // Save negated node for later, and add new node to CSG tree
    NodeId const negated = n.node;
    auto&& [node_id, inserted] = ub_->insert_csg(std::move(n));

    if (!md.empty())
    {
        // Metadata may be empty if the negated region is part of a subtraction
        // or RDV
        ub_->insert_md(node_id, std::move(md));
    }

    // Create bounding zone and add region
    BoundingZone bz = ub_->bounds(negated);
    bz.negate();
    ub_->insert_region(node_id, std::move(bz), transforms_.back());

    return node_id;
}

//---------------------------------------------------------------------------//
/*!
 * Apply a transform within this scope.
 */
[[nodiscard]] PopVBTransformOnDestruct
VolumeBuilder::make_scoped_transform(VariantTransform const& t)
{
    // Apply the current local transform to get a new local transform, and add
    // it to the stack
    this->push_transform(apply_transform(this->local_transform(), t));

    // Return the helper class that will pop the last transform
    CELER_ENSURE(transforms_.size() > 1);
    return PopVBTransformOnDestruct(this);
}

//---------------------------------------------------------------------------//
// PRIVATE METHODS
//---------------------------------------------------------------------------//
/*!
 * Add a new variant transform.
 */
void VolumeBuilder::push_transform(VariantTransform&& vt)
{
    auto trans_id = ub_->insert_transform(std::move(vt));
    transforms_.push_back(trans_id);
}

//---------------------------------------------------------------------------//
/*!
 * Pop the last transform, used only by PopVBTransformOnDestruct.
 */
void VolumeBuilder::pop_transform()
{
    CELER_EXPECT(transforms_.size() > 1);
    transforms_.pop_back();
    CELER_ENSURE(!transforms_.empty());
}

//---------------------------------------------------------------------------//
/*!
 * Construct with a volume builder pointer.
 */
PopVBTransformOnDestruct::PopVBTransformOnDestruct(VolumeBuilder* vb) : vb_{vb}
{
    CELER_EXPECT(vb_);
    CELER_EXPECT(vb_->transforms_.size() > 1);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
