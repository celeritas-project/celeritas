//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/alongstep/AlongStepRZMapFieldMscAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/em/data/UrbanMscData.hh"
#include "celeritas/field/RZMapFieldData.hh"
#include "celeritas/field/RZMapFieldInput.hh"
#include "celeritas/field/RZMapFieldParams.hh"
#include "celeritas/global/ActionInterface.hh"

namespace celeritas
{
class UrbanMscParams;
class PhysicsParams;
class MaterialParams;
class ParticleParams;

//---------------------------------------------------------------------------//
/*!
 * Along-step kernel with optional MSC and a RZMapField.
 */
class AlongStepRZMapFieldMscAction final : public ExplicitActionInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstMsc = std::shared_ptr<UrbanMscParams const>;
    using SPConstFieldParams = std::shared_ptr<RZMapFieldParams const>;
    //!@}

  public:
    // Construct with next action ID, optional MSC, magnetic field
    AlongStepRZMapFieldMscAction(ActionId id,
                                 RZMapFieldInput const& input,
                                 SPConstMsc msc);

    // Default destructor
    ~AlongStepRZMapFieldMscAction();

    // Launch kernel with host data
    void execute(ParamsHostCRef const&, StateHostRef&) const final;

    // Launch kernel with device data
    void execute(ParamsDeviceCRef const&, StateDeviceRef&) const final;

    //! ID of the model
    ActionId action_id() const final { return id_; }

    //! Short name for the interaction kernel
    std::string label() const final { return "along-step-mapfield-msc"; }

    //! Name of the model, for user interaction
    std::string description() const final
    {
        return "along-step in a r-z map field with Urban MSC";
    }

    //! Dependency ordering of the action
    ActionOrder order() const final { return ActionOrder::along; }

    //// ACCESSORS ////

    //! Whether MSC is in use
    bool has_msc() const { return static_cast<bool>(msc_); }

    //! Field map data
    SPConstFieldParams field() const { return field_; }

  private:
    ActionId id_;
    SPConstMsc msc_;
    SPConstFieldParams field_;

    // TODO: kind of hacky way to support msc being optional
    // (required because we have to pass "empty" refs if they're missing)
    template<MemSpace M>
    struct ExternalRefs
    {
        UrbanMscData<Ownership::const_reference, M> msc;

        ExternalRefs(SPConstMsc const& msc_params);
    };

    ExternalRefs<MemSpace::host> host_data_;
    ExternalRefs<MemSpace::device> device_data_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//

#if !CELER_USE_DEVICE
inline void AlongStepRZMapFieldMscAction::execute(ParamsDeviceCRef const&,
                                                  StateDeviceRef&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
