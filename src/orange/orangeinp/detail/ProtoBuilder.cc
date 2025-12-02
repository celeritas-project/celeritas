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
    , num_univs_{protos_.size()}
{
    CELER_EXPECT(inp_);
    CELER_EXPECT(opts.tol);

    inp_->tol = opts.tol;
    inp_->universes.resize(protos_.size());
}

//---------------------------------------------------------------------------//
/*!
 * Save debugging data for a universe.
 */
void ProtoBuilder::save_json(JsonPimpl&& jp) const
{
    CELER_EXPECT(this->save_json());
    save_json_(this->current_uid(), std::move(jp));
}

//---------------------------------------------------------------------------//
/*!
 * Add a universe to the input.
 *
 * This may be called *once* per proto.
 */
void ProtoBuilder::insert(VariantUniverseInput&& unit)
{
    inp_->universes[this->current_uid().get()] = std::move(unit);

    if (!this->is_global_universe())
    {
        ++num_univs_inserted_;
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
