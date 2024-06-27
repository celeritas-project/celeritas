//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/DummyAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <vector>

#include "corecel/Types.hh"
#include "corecel/data/AuxInterface.hh"
#include "celeritas/global/ActionInterface.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
struct DummyState : public AuxStateInterface
{
    MemSpace memspace{MemSpace::size_};
    StreamId stream_id{};
    size_type size{};
    std::vector<std::string> action_order;
};

//---------------------------------------------------------------------------//
class DummyParams final : public AuxParamsInterface
{
  public:
    //! Construct with aux ID
    explicit DummyParams(AuxId aux_id) : aux_id_{aux_id} {}

    //! Index of this class instance in its registry
    AuxId aux_id() const final { return aux_id_; }

    //! Label for the auxiliary data
    std::string_view label() const final { return "dummy-params"; }

    //! Build state data for a stream
    UPState create_state(MemSpace m, StreamId id, size_type size) const final;

  private:
    AuxId aux_id_;
};

//---------------------------------------------------------------------------//
class DummyAction final : public ExplicitCoreActionInterface,
                          public ConcreteAction
{
  public:
    DummyAction(ActionId id, ActionOrder order, std::string&& label, AuxId aux);

    void execute(CoreParams const&, CoreStateHost& state) const final;
    void execute(CoreParams const&, CoreStateDevice& state) const final;

    ActionOrder order() const final { return order_; }

  private:
    ActionOrder order_;
    AuxId aux_id_;

    // Add the action order
    void execute_impl(DummyState& state) const;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
