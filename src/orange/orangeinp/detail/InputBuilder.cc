//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/InputBuilder.cc
//---------------------------------------------------------------------------//
#include "InputBuilder.hh"

#include "orange/BoundingBoxUtils.hh"

#include "../ProtoInterface.hh"

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with output pointer, geometry construction options, and protos.
 */
InputBuilder::InputBuilder(OrangeInput* inp,
                           Tol const& tol,
                           ProtoMap const& protos)
    : inp_{inp}, protos_{protos}, bboxes_{protos_.size()}
{
    CELER_EXPECT(inp_);
    CELER_EXPECT(tol);

    inp_->tol = tol;
    inp_->universes.reserve(protos_.size());
}

//---------------------------------------------------------------------------//
/*!
 * Expand the bounding box of a universe.
 *
 * Creating successive instances of a universe's "parent" volume will expand
 * the possible extents of that universe. In SCALE, same universe could be
 * "holed" (placed) in different volumes with different bounds, so long as the
 * enclosures are within the extents of the child universe.
 */
void InputBuilder::expand_bbox(UniverseId uid, BBox const& local_bbox)
{
    CELER_EXPECT(uid < bboxes_.size());
    BBox& target = bboxes_[uid.get()];
    target = calc_union(target, local_bbox);
}

//---------------------------------------------------------------------------//
/*!
 * Add a universe to the input.
 *
 * This may be called *once* per proto.
 */
void InputBuilder::insert(VariantUniverseInput&& unit)
{
    CELER_EXPECT(inp_->universes.size() < protos_.size());

    inp_->universes.emplace_back(std::move(unit));
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
