//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/DummyAction.cc
//---------------------------------------------------------------------------//
#include "DummyAction.hh"

#include <memory>

#include "corecel/data/AuxStateVec.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// Build state data for a stream
auto DummyParams::create_state(MemSpace m,
                               StreamId stream_id,
                               size_type size) const -> UPState
{
    auto result = std::make_unique<DummyState>();
    result->memspace = m;
    result->stream_id = stream_id;
    result->size = size;
    return result;
}

//---------------------------------------------------------------------------//

DummyAction::DummyAction(ActionId id,
                         ActionOrder order,
                         std::string&& label,
                         AuxId aux)
    : ConcreteAction{id, std::move(label)}, order_{order}, aux_id_{aux}
{
}

void DummyAction::execute(CoreParams const&, CoreStateHost& state) const
{
    return this->execute_impl(get<DummyState>(state.aux(), aux_id_));
}

void DummyAction::execute(CoreParams const&, CoreStateDevice& state) const
{
    return this->execute_impl(get<DummyState>(state.aux(), aux_id_));
}

void DummyAction::execute_impl(DummyState& state) const
{
    state.action_order.emplace_back(to_cstring(order_));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
