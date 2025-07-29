//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/OpticalLaunchAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string_view>

#include "corecel/Macros.hh"
#include "corecel/data/AuxInterface.hh"
#include "corecel/data/AuxStateVec.hh"
#include "corecel/math/NumericLimits.hh"
#include "celeritas/global/ActionInterface.hh"

#include "../Model.hh"

namespace celeritas
{
class CoreParams;

namespace optical
{
class CoreParams;
template<MemSpace M>
class CoreState;
class Transporter;
}  // namespace optical

namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Manage optical params and state, launching the optical stepping loop.
 *
 * This stores the optical tracking loop's core params, initializing them at
 * the beginning of the run, and stores the optical core state as "aux" data.
 */
class OpticalLaunchAction : public AuxParamsInterface,
                            public CoreStepActionInterface,
                            public CoreBeginRunActionInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPOpticalParams = std::shared_ptr<optical::CoreParams>;
    //!@}

    struct Input
    {
        SPOpticalParams optical_params;
        size_type num_track_slots{};
        size_type max_step_iters{numeric_limits<size_type>::max()};
        size_type auto_flush{};

        //! True if all input is assigned and valid
        explicit operator bool() const
        {
            return optical_params && num_track_slots > 0 && auto_flush > 0;
        }
    };

  public:
    // Construct and add to core params
    static std::shared_ptr<OpticalLaunchAction>
    make_and_insert(CoreParams const&, Input&&);

    // Construct with IDs and optical params
    OpticalLaunchAction(ActionId, AuxId, Input&&);

    //!@{
    //! \name Aux/action metadata interface

    //! Short name for the action
    std::string_view label() const final { return "optical-offload-launch"; }
    // Name of the action (for user output)
    std::string_view description() const final;
    //!@}

    //!@{
    //! \name Aux interface

    //! Index of this class instance in its registry
    AuxId aux_id() const final { return aux_id_; }
    // Build optical core state data for a stream
    UPState create_state(MemSpace, StreamId, size_type) const final;
    //!@}

    //!@{
    //! \name BeginRunAction interface

    // Create the action groups and get a pointer to the aux data
    void begin_run(CoreParams const&, CoreStateHost&) final;
    // Create the action groups and get a pointer to the aux data
    void begin_run(CoreParams const&, CoreStateDevice&) final;
    //!@}

    //!@{
    //! \name Action interface

    //! ID of the model
    ActionId action_id() const final { return action_id_; }
    //! Dependency ordering of the action
    StepActionOrder order() const final { return StepActionOrder::end; }
    // Launch kernel with host data
    void step(CoreParams const&, CoreStateHost&) const final;
    // Launch kernel with device data
    void step(CoreParams const&, CoreStateDevice&) const final;
    //!@}

    // TODO: local end event to flush initializers??

    //!@{
    //! \name Accessors

    //! Optical tracks per stream
    size_type state_size() const { return data_.num_track_slots; }
    //! Optical core params
    optical::CoreParams const& optical_params() const
    {
        return *data_.optical_params;
    }
    //!@}

  private:
    using SPTransporter = std::shared_ptr<optical::Transporter>;

    //// DATA ////

    ActionId action_id_;
    AuxId aux_id_;
    Input data_;
    SPTransporter transport_;

    //// HELPERS ////

    template<MemSpace M>
    void execute_impl(CoreParams const&, CoreState<M>&) const;
    template<MemSpace M>
    void begin_run_impl(CoreState<M>&);
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
