//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/DummyAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/global/ActionInterface.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
class DummyAction final : public ExplicitActionInterface, public ConcreteAction
{
  public:
    // Construct with ID and label
    using ConcreteAction::ConcreteAction;

    void execute(CoreParams const&, CoreStateHost&) const final
    {
        ++num_execute_host_;
    }
    void execute(CoreParams const&, CoreStateDevice&) const final
    {
        ++num_execute_device_;
    }

    int num_execute_host() const { return num_execute_host_; }
    int num_execute_device() const { return num_execute_device_; }

    ActionOrder order() const final { return ActionOrder::post_post; }

  private:
    mutable int num_execute_host_{0};
    mutable int num_execute_device_{0};
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
