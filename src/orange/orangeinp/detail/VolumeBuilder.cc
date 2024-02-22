//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
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
 * Access the local-to-global transform during construction.
 */
VariantTransform const& VolumeBuilder::local_transform() const
{
    CELER_EXPECT(!transforms_.empty());
    return ub_->transform(transforms_.back());
}

//-------------------------------------------------------------------------//
/*!
 * Add a region to the CSG tree.
 */
NodeId
VolumeBuilder::insert_region(Metadata&& md, Joined&& j, BoundingZone&& bzone)
{
    auto node_id = ub_->insert_csg(std::move(j)).first;
    CELER_ASSERT(!transforms_.empty());
    ub_->insert_region(node_id, std::move(bzone), transforms_.back());

    // Always add metadata
    ub_->insert_md(node_id, std::move(md));

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
