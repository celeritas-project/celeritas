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
#include "celeritas/global/ActionInterface.hh"

#include "../Model.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
template<class P, template<MemSpace M> class S>
class ActionGroups;
class CoreParams;

namespace optical
{
class CoreParams;
template<MemSpace M>
class CoreState;
class MaterialParams;
}  // namespace optical

namespace detail
{
class OffloadParams;
}

namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Manage optical params and state, launching the optical stepping loop.
 *
 * This stores the optical tracking loop's core params, initializing them at
 * the beginning of the run, and stores the optical core state as "aux"
 * data.
 */
class OpticalLaunchAction : public AuxParamsInterface,
                            public CoreStepActionInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPOffloadParams = std::shared_ptr<detail::OffloadParams>;
    using SPConstMaterial = std::shared_ptr<optical::MaterialParams const>;
    //!@}

    struct Input
    {
        SPConstMaterial material;
        std::vector<optical::Model::ModelBuilder> model_builders;
        SPOffloadParams offload;
        size_type num_track_slots{};
        size_type initializer_capacity{};

        //! True if all input is assigned and valid
        explicit operator bool() const
        {
            return material && offload && num_track_slots > 0
                   && initializer_capacity > 0;
        }
    };

  public:
    // Construct and add to core params
    static std::shared_ptr<OpticalLaunchAction>
    make_and_insert(CoreParams const&, Input&&);

    // Construct with IDs, core for copying params, offload gen data
    OpticalLaunchAction(ActionId, AuxId, CoreParams const&, Input&&);

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
    //! \name Action interface

    //! ID of the model
    ActionId action_id() const final { return action_id_; }
    //! Dependency ordering of the action
    StepActionOrder order() const final { return StepActionOrder::user_post; }
    // Launch kernel with host data
    void step(CoreParams const&, CoreStateHost&) const final;
    // Launch kernel with device data
    void step(CoreParams const&, CoreStateDevice&) const final;
    //!@}

    // TODO: local end event to flush initializers??

    //!@{
    //! \name Accessors

    //! Optical tracks per stream
    size_type state_size() const { return state_size_; }
    //! Optical core params
    optical::CoreParams const& optical_params() const
    {
        return *optical_params_;
    }
    //! Offload params
    detail::OffloadParams const& offload_params() const
    {
        return *offload_params_;
    }

    //!@}

  private:
    using ActionGroupsT = ActionGroups<optical::CoreParams, optical::CoreState>;
    using SPOpticalParams = std::shared_ptr<optical::CoreParams>;
    using SPActionGroups = std::shared_ptr<ActionGroupsT>;

    //// DATA ////

    ActionId action_id_;
    AuxId aux_id_;
    SPOffloadParams offload_params_;
    SPOpticalParams optical_params_;
    SPActionGroups optical_actions_;
    size_type state_size_;

    //// HELPERS ////

    template<MemSpace M>
    void execute_impl(CoreParams const&, CoreState<M>&) const;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
