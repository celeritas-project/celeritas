//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/OffloadGatherAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "corecel/Macros.hh"
#include "corecel/data/AuxInterface.hh"
#include "corecel/data/Collection.hh"
#include "celeritas/global/ActionInterface.hh"

#include "OffloadData.hh"

namespace celeritas
{
class CoreParams;

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
class OffloadGatherAction final : public CoreStepActionInterface,
                                  public AuxParamsInterface
{
  public:
    // Construct and add to core params
    static std::shared_ptr<OffloadGatherAction>
    make_and_insert(CoreParams const&);

    // Construct with action ID and storage
    OffloadGatherAction(ActionId id, AuxId aux_id);

    //!@{
    //! \name Aux interface

    //! Index of this class instance in its registry
    AuxId aux_id() const final { return aux_id_; }
    // Build state data for a stream
    UPState create_state(MemSpace, StreamId, size_type) const final;
    //!@}

    //!@{
    //! \name Action interface

    //! ID of the model
    ActionId action_id() const final { return action_id_; }
    //! Short name for the action
    std::string_view label() const final { return "optical-offload-gather"; }
    // Name of the action (for user output)
    std::string_view description() const final;
    //!@}

    //!@{
    //! \name StepAction interface

    //! Dependency ordering of the action
    StepActionOrder order() const final { return StepActionOrder::user_pre; }
    // Launch kernel with host data
    void step(CoreParams const&, CoreStateHost&) const final;
    // Launch kernel with device data
    void step(CoreParams const&, CoreStateDevice&) const final;
    //!@}

  private:
    //// DATA ////

    ActionId action_id_;
    AuxId aux_id_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
