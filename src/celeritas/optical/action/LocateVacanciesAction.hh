//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/LocateVacanciesAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include "ActionInterface.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Find the vacant track slots at the end of the step.
 *
 * \todo Rename?
 */
class LocateVacanciesAction final : public OpticalStepActionInterface,
                                    public ConcreteAction
{
  public:
    //! Construct with ID
    explicit LocateVacanciesAction(ActionId);

    //! Execute the action with host data
    void step(CoreParams const&, CoreStateHost&) const final;

    //! Execute the action with device data
    void step(CoreParams const&, CoreStateDevice&) const final;

    //! Dependency ordering of the action
    StepActionOrder order() const final { return StepActionOrder::end; }

  private:
    template<MemSpace M>
    void step_impl(CoreState<M>&) const;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
