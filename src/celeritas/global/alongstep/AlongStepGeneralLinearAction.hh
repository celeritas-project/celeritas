//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/alongstep/AlongStepGeneralLinearAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/em/data/FluctuationData.hh"
#include "celeritas/em/data/UrbanMscData.hh"
#include "celeritas/global/ActionInterface.hh"
#include "celeritas/global/CoreTrackData.hh"

namespace celeritas
{
class UrbanMscModel;
class FluctuationParams;

class PhysicsParams;
class MaterialParams;
class ParticleParams;
class ActionManager;

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
    using SPConstFluctuations = std::shared_ptr<const FluctuationParams>;
    using SPConstMsc          = std::shared_ptr<const UrbanMscModel>;
    //!@}

  public:
    static std::shared_ptr<AlongStepGeneralLinearAction>
    from_params(const MaterialParams& materials,
                const ParticleParams& particles,
                const PhysicsParams&  physics,
                bool                  eloss_fluctuation,
                ActionManager*        actions);

    // Construct with next action ID, and optional EM energy fluctuation
    AlongStepGeneralLinearAction(ActionId            id,
                                 SPConstFluctuations fluct,
                                 SPConstMsc          msc);

    // Default destructor
    ~AlongStepGeneralLinearAction();

    // Launch kernel with host data
    void execute(CoreHostRef const&) const final;

    // Launch kernel with device data
    void execute(CoreDeviceRef const&) const final;

    //! ID of the model
    ActionId action_id() const final { return id_; }

    //! Short name for the interaction kernel
    std::string label() const final { return "along-step-general-linear"; }

    //! Name of the model, for user interaction
    std::string description() const final
    {
        return "along-step for particles with no field";
    }

    //// ACCESSORS ////

    //! Whether energy flucutation is in use
    bool has_fluct() const { return static_cast<bool>(fluct_); }

    //! Whether MSC is in use
    bool has_msc() const { return static_cast<bool>(msc_); }

  private:
    ActionId            id_;
    SPConstFluctuations fluct_;
    SPConstMsc          msc_;

    // TODO: kind of hacky way to support fluct/msc being optional
    // (required because we have to pass "empty" refs if they're missing)
    template<MemSpace M>
    struct ExternalRefs
    {
        FluctuationData<Ownership::const_reference, M> fluct;
        UrbanMscData<Ownership::const_reference, M>    msc;

        ExternalRefs(const SPConstFluctuations& fluct_params,
                     const SPConstMsc&          msc_params);
    };

    ExternalRefs<MemSpace::host>   host_data_;
    ExternalRefs<MemSpace::device> device_data_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//

#if !CELER_USE_DEVICE
inline void AlongStepGeneralLinearAction::execute(CoreDeviceRef const&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
} // namespace celeritas
