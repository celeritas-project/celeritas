//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
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

namespace celeritas
{
//---------------------------------------------------------------------------//
class CoreParams;
namespace optical
{
class CoreParams;
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

  public:
    // Construct and add to core params
    static std::shared_ptr<OpticalLaunchAction>
    make_and_insert(CoreParams const& core,
                    SPConstMaterial material,
                    SPOffloadParams offload);

    // Construct with IDs, core for copying params, offload gen data
    OpticalLaunchAction(ActionId id,
                        AuxId data_id,
                        CoreParams const& core,
                        SPConstMaterial material,
                        SPOffloadParams offload);

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

  private:
    using SPOpticalParams = std::shared_ptr<optical::CoreParams>;

    //// DATA ////

    ActionId action_id_;
    AuxId aux_id_;
    SPOffloadParams offload_params_;
    SPOpticalParams optical_params_;

    //// HELPERS ////

    template<MemSpace M>
    void execute_impl(CoreParams const&, CoreState<M>&) const;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
