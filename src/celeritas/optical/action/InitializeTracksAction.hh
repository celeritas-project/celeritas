//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/InitializeTracksAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include "ActionInterface.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Initialize optical track states.
 *
 * If there are more empty slots than new secondaries, they will be filled by
 * any track initializers remaining from previous steps using the position.
 */
class InitializeTracksAction final : public OpticalStepActionInterface,
                                     public ConcreteAction
{
  public:
    //! Construct with ID
    explicit InitializeTracksAction(ActionId);

    //! Execute the action with host data
    void step(CoreParams const&, CoreStateHost&) const final;

    //! Execute the action with device data
    void step(CoreParams const&, CoreStateDevice&) const final;

    //! Dependency ordering of the action
    StepActionOrder order() const final { return StepActionOrder::start; }

  private:
    template<MemSpace M>
    void step_impl(CoreParams const&, CoreState<M>&) const;

    void step_impl(CoreParams const&, CoreStateHost&, size_type) const;
    void step_impl(CoreParams const&, CoreStateDevice&, size_type) const;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
