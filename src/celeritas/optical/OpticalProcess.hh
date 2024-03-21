//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalProcess.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/global/ActionInterface.hh"

namespace celeritas
{
class ValueGridBuilder;

//---------------------------------------------------------------------------//
/*!
 * Brief class description.
 *
 * Optional detailed class description, and possibly example usage:
 * \code
    OpticalProcess ...;
   \endcode
 */
class OpticalProcess : public ExplicitActionInterface
{
  public:
    //!@{
    //! \name Type aliases
    using UPConstGridBuilder = std::unique_ptr<ValueGridBuilder const>;
    using StepLimitBuilder = UPConstGridBuilder;
    using ActionIdIter = RangeIter<ActionId>;
    using OpticalXsBuilders = std::vector<UPConstGridBuilder>;
    //!@}

  public:
    // Virtual destructor for polymorphic deletion
    virtual ~OpticalProcess();

    //! Get the interaction cross sections for optical photons
    virtual XsStepLimitBuilder step_limits() const = 0;

    //! Dependency ordering of the action
    ActionOrder order() const override final { return ActionOrder::post; }

    //! ID of this action for verification
    ActionId action_id() const override final { return action_id_; }

  protected:
    //! Require action id to be populated by subclasses
    OpticalProcess(ActionId id)
        : action_id_(id)
    {}

    ActionId action_id_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
