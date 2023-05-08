//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/ExtendFromSecondariesAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/global/ActionInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Create track initializers on device from secondary particles.
 *
 * \sa celeritas::extend_from_secondaries
 */
class ExtendFromSecondariesAction final : public ExplicitActionInterface
{
  public:
    //! Construct with explicit Id
    explicit ExtendFromSecondariesAction(ActionId id) : id_(id) {}

    //! Default destructor
    ~ExtendFromSecondariesAction() = default;

    // Execute the action with host data
    void execute(CoreParams const& params, CoreStateHost& state) const final;

    // Execute the action with device data
    void execute(CoreParams const& params, CoreStateDevice& state) const final;

    //! ID of the action
    ActionId action_id() const final { return id_; }

    //! Short name for the action
    std::string label() const final { return "extend-from-secondaries"; }

    //! Description of the action for user interaction
    std::string description() const final
    {
        return "create secondary track initializers";
    }

    //! Dependency ordering of the action
    ActionOrder order() const final { return ActionOrder::end; }

  private:
    ActionId id_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
