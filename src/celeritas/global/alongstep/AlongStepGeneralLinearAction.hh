//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/alongstep/AlongStepGeneralLinearAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/Types.hh"
#include "celeritas/em/data/FluctuationData.hh"
#include "celeritas/em/data/UrbanMscData.hh"
#include "celeritas/global/ActionInterface.hh"

namespace celeritas
{
class UrbanMscParams;
class FluctuationParams;
class PhysicsParams;
class MaterialParams;
class ParticleParams;

//---------------------------------------------------------------------------//
/*!
 * Along-step kernel for particles without fields.
 *
 * This kernel is for problems without EM fields, for particle types that may
 * have (but do not *need* to have) along-step energy loss, optional energy
 * fluctuation, and optional multiple scattering.
 */
class AlongStepGeneralLinearAction final : public ExplicitActionInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstFluctuations = std::shared_ptr<FluctuationParams const>;
    using SPConstMsc = std::shared_ptr<UrbanMscParams const>;
    //!@}

  public:
    static std::shared_ptr<AlongStepGeneralLinearAction>
    from_params(ActionId id,
                MaterialParams const& materials,
                ParticleParams const& particles,
                SPConstMsc const& msc,
                bool eloss_fluctuation);

    // Construct with next action ID, and optional EM energy fluctuation
    AlongStepGeneralLinearAction(ActionId id,
                                 SPConstFluctuations fluct,
                                 SPConstMsc msc);

    // Default destructor
    ~AlongStepGeneralLinearAction();

    // Launch kernel with host data
    void execute(CoreParams const&, StateHostRef&) const final;

    // Launch kernel with device data
    void execute(CoreParams const&, StateDeviceRef&) const final;

    //! ID of the model
    ActionId action_id() const final { return id_; }

    //! Short name for the along-step kernel
    std::string label() const final { return "along-step-general-linear"; }

    //! Name of the model, for user interaction
    std::string description() const final
    {
        return "along-step for particles with no field";
    }

    //! Dependency ordering of the action
    ActionOrder order() const final { return ActionOrder::along; }

    //// ACCESSORS ////

    //! Whether energy flucutation is in use
    bool has_fluct() const { return static_cast<bool>(fluct_); }

    //! Whether MSC is in use
    bool has_msc() const { return static_cast<bool>(msc_); }

  private:
    ActionId id_;
    SPConstFluctuations fluct_;
    SPConstMsc msc_;

    // TODO: kind of hacky way to support fluct/msc being optional
    // (required because we have to pass "empty" refs if they're missing)
    template<MemSpace M>
    struct ExternalRefs
    {
        FluctuationData<Ownership::const_reference, M> fluct;
        UrbanMscData<Ownership::const_reference, M> msc;

        ExternalRefs(SPConstFluctuations const& fluct_params,
                     SPConstMsc const& msc_params);
    };

    ExternalRefs<MemSpace::host> host_data_;
    ExternalRefs<MemSpace::device> device_data_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//

#if !CELER_USE_DEVICE
inline void
AlongStepGeneralLinearAction::execute(CoreParams const&, StateDeviceRef&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
