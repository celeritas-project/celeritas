//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/OffloadGatherAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "corecel/Macros.hh"
#include "corecel/data/AuxInterface.hh"
#include "corecel/data/Collection.hh"
#include "celeritas/global/ActionInterface.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Collect pre-step data needed to generate optical distribution data.
 *
 * This pre-step action stores the optical material ID and other
 * beginning-of-step properties so that optical photons can be generated
 * between the start and end points of the step.
 *
 * \sa OffloadGatherExecutor
 */
class OffloadGatherAction final : public CoreStepActionInterface
{
  public:
    // Construct with action ID and storage
    OffloadGatherAction(ActionId id, AuxId data_id);

    // Launch kernel with host data
    void step(CoreParams const&, CoreStateHost&) const final;

    // Launch kernel with device data
    void step(CoreParams const&, CoreStateDevice&) const final;

    //! ID of the model
    ActionId action_id() const final { return id_; }

    //! Short name for the action
    std::string_view label() const final { return "optical-offload-gather"; }

    // Name of the action (for user output)
    std::string_view description() const final;

    //! Dependency ordering of the action
    StepActionOrder order() const final { return StepActionOrder::user_pre; }

  private:
    //// DATA ////

    ActionId id_;
    AuxId data_id_;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
