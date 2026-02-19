//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/StatusChecker.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string_view>

#include "corecel/Types.hh"
#include "corecel/data/AuxParams.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "corecel/data/ParamsDataStore.hh"
#include "celeritas/Types.hh"
#include "celeritas/global/ActionInterface.hh"

#include "StatusCheckData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class ActionRegistry;
class CoreParams;
template<MemSpace>
class CoreState;

//---------------------------------------------------------------------------//
/*!
 * Verify a consistent simulation state after performing an action.
 *
 * This is used as a debug option in the step executor to check that actions
 * and simulation state are consistent.
 *
 * Since this is called manually by the stepper, multiple times per step, it is
 * \em not a CoreStepAction. It's meant to be used inside the \c
 * ActionSequence itself, called after every action.
 */
class StatusChecker final
    : public CoreBeginRunActionInterface,
      public AuxParams<StatusCheckParamsData, StatusCheckStateData>
{
  public:
    // Construct and add to core params
    static std::shared_ptr<StatusChecker>
    make_and_insert(CoreParams const& core);

    // Construct with IDs
    StatusChecker(ActionId action_id, AuxId aux_id);

    //!@{
    //! \name Aux/action metadata interface

    //! Label for the auxiliary data and action
    std::string_view label() const final { return "status-check"; }
    // Description of the action
    std::string_view description() const final;
    //!@}

    //!@{
    //! \name Aux params interface

    //! Index of this class instance in its registry
    AuxId aux_id() const final { return aux_id_; }
    //!@}

    //!@{
    //! \name Begin run interface

    //! Index of this class instance in its registry
    ActionId action_id() const final { return action_id_; }
    // Set host data at the beginning of a run
    void begin_run(CoreParams const&, CoreStateHost&) final;
    // Set device data at the beginning of a run
    void begin_run(CoreParams const&, CoreStateDevice&) final;
    //!@}

    //!@{
    //! \name Data interface

    //! Access data on host
    HostRef const& host_ref() const final { return data_.host_ref(); }
    //! Access data on device
    DeviceRef const& device_ref() const final { return data_.device_ref(); }
    //!@}

    // Execute *manually* with the last action's ID and the state
    template<MemSpace M>
    void
    step(ActionId prev_action, CoreParams const&, CoreState<M>& state) const;

  private:
    template<MemSpace M>
    using StatusStateRef = StatusCheckStateData<Ownership::reference, M>;

    //// DATA ////

    ActionId action_id_;
    AuxId aux_id_;
    ParamsDataStore<StatusCheckParamsData> data_;

    //// HELPER FUNCTIONS ////

    void begin_run_impl(CoreParams const&);

    void step_impl(CoreParams const&,
                   CoreState<MemSpace::host>&,
                   StatusStateRef<MemSpace::host> const&) const;
    void step_impl(CoreParams const&,
                   CoreState<MemSpace::device>&,
                   StatusStateRef<MemSpace::device> const&) const;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
