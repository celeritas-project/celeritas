//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/alongstep/AlongStepGeneralLinearAction.cc
//---------------------------------------------------------------------------//
#include "AlongStepGeneralLinearAction.hh"

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/data/Ref.hh"
#include "corecel/sys/MultiExceptionHandler.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/em/FluctuationParams.hh"
#include "celeritas/em/model/UrbanMscModel.hh"
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/global/alongstep/detail/AlongStepLauncherImpl.hh"
#include "celeritas/phys/PhysicsParams.hh"

#include "AlongStepLauncher.hh"
#include "detail/AlongStepGeneralLinear.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct the along-step action from input parameters.
 */
std::shared_ptr<AlongStepGeneralLinearAction>
AlongStepGeneralLinearAction::from_params(ActionId              id,
                                          const MaterialParams& materials,
                                          const ParticleParams& particles,
                                          const PhysicsParams&  physics,
                                          bool eloss_fluctuation)
{
    SPConstFluctuations fluct;
    if (eloss_fluctuation)
    {
        fluct = std::make_shared<FluctuationParams>(particles, materials);
    }

    // Super hacky!! This will be cleaned up later.
    SPConstMsc msc;
    for (auto mid : range(ModelId{physics.num_models()}))
    {
        msc = std::dynamic_pointer_cast<const UrbanMscModel>(
            physics.model(mid));
        if (msc)
        {
            // Found MSC
            break;
        }
    }

    return std::make_shared<AlongStepGeneralLinearAction>(
        id, std::move(fluct), std::move(msc));
}

//---------------------------------------------------------------------------//
/*!
 * Construct with next action ID and optional energy loss parameters.
 */
AlongStepGeneralLinearAction::AlongStepGeneralLinearAction(
    ActionId id, SPConstFluctuations fluct, SPConstMsc msc)
    : id_(id)
    , fluct_(std::move(fluct))
    , msc_(std::move(msc))
    , host_data_(fluct_, msc_)
    , device_data_(fluct_, msc_)
{
    CELER_EXPECT(id_);
}

//---------------------------------------------------------------------------//
//! Default destructor
AlongStepGeneralLinearAction::~AlongStepGeneralLinearAction() = default;

//---------------------------------------------------------------------------//
/*!
 * Launch the along-step action on host.
 */
void AlongStepGeneralLinearAction::execute(CoreHostRef const& data) const
{
    CELER_EXPECT(data);

    MultiExceptionHandler capture_exception;
    auto launch = make_along_step_launcher(data,
                                           host_data_.msc,
                                           NoData{},
                                           host_data_.fluct,
                                           detail::along_step_general_linear);

#pragma omp parallel for
    for (size_type i = 0; i < data.states.size(); ++i)
    {
        CELER_TRY_ELSE(launch(ThreadId{i}), capture_exception);
    }
    log_and_rethrow(std::move(capture_exception));
}

//---------------------------------------------------------------------------//
/*!
 * Save references from host/device data.
 */
template<MemSpace M>
AlongStepGeneralLinearAction::ExternalRefs<M>::ExternalRefs(
    const SPConstFluctuations& fluct_params, const SPConstMsc& msc_params)
{
    if (M == MemSpace::device && !celeritas::device())
    {
        // Skip device copy if disabled
        return;
    }

    if (fluct_params)
    {
        fluct = get_ref<M>(*fluct_params);
    }
    if (msc_params)
    {
        msc = get_ref<M>(*msc_params);
    }
}

//---------------------------------------------------------------------------//
} // namespace celeritas
