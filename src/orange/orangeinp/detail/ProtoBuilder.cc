//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/ProtoBuilder.cc
//---------------------------------------------------------------------------//
#include "ProtoBuilder.hh"

#include "corecel/io/JsonPimpl.hh"
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
ProtoBuilder::ProtoBuilder(OrangeInput* inp,
                           ProtoMap const& protos,
                           Options const& opts)
    : inp_{inp}
    , protos_{protos}
    , save_json_{opts.save_json}
    , bboxes_{protos_.size()}
{
    CELER_EXPECT(inp_);
    CELER_EXPECT(opts.tol);

    inp_->tol = opts.tol;
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
void ProtoBuilder::expand_bbox(UniverseId univ_id, BBox const& local_bbox)
{
    CELER_EXPECT(univ_id < bboxes_.size());
    BBox& target = bboxes_[univ_id.get()];
    target = calc_union(target, local_bbox);
}

//---------------------------------------------------------------------------//
/*!
 * Save debugging data for a universe.
 */
void ProtoBuilder::save_json(JsonPimpl&& jp) const
{
    CELER_EXPECT(this->save_json());
    CELER_EXPECT(inp_->universes.size() < protos_.size());

    save_json_(UniverseId(inp_->universes.size()), std::move(jp));
}

//---------------------------------------------------------------------------//
/*!
 * Add a universe to the input.
 *
 * This may be called *once* per proto.
 */
void ProtoBuilder::insert(VariantUniverseInput&& unit)
{
    CELER_EXPECT(inp_->universes.size() < protos_.size());

    inp_->universes.emplace_back(std::move(unit));
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
