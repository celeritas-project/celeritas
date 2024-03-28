//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/InputBuilder.cc
//---------------------------------------------------------------------------//
#include "InputBuilder.hh"

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
    : inp_{inp}, protos_{protos}

{
    CELER_EXPECT(inp_);
    CELER_EXPECT(tol);

    inp_->tol = tol;
    inp_->universes.reserve(protos_.size());
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
