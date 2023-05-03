//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/alongstep/AlongStepGeneralLinearAction.cc
//---------------------------------------------------------------------------//
#include "AlongStepGeneralLinearAction.hh"

#include <utility>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/Ref.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/MultiExceptionHandler.hh"
#include "celeritas/Types.hh"
#include "celeritas/em/FluctuationParams.hh"
#include "celeritas/em/UrbanMscParams.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/global/KernelContextException.hh"
#include "celeritas/global/TrackLauncher.hh"
#include "celeritas/phys/PhysicsParams.hh"

#include "detail/AlongStepGeneralLinear.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct the along-step action from input parameters.
 */
std::shared_ptr<AlongStepGeneralLinearAction>
AlongStepGeneralLinearAction::from_params(ActionId id,
                                          MaterialParams const& materials,
                                          ParticleParams const& particles,
                                          SPConstMsc const& msc,
                                          bool eloss_fluctuation)
{
    SPConstFluctuations fluct;
    if (eloss_fluctuation)
    {
        fluct = std::make_shared<FluctuationParams>(particles, materials);
    }

    return std::make_shared<AlongStepGeneralLinearAction>(
        id, std::move(fluct), msc);
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
void AlongStepGeneralLinearAction::execute(CoreParams const& params,
                                           CoreStateHost& state) const
{
    MultiExceptionHandler capture_exception;
    auto launch
        = make_along_step_track_launcher(*params.ptr<MemSpace::native>(),
                                         *state.ptr(),
                                         this->action_id(),
                                         detail::along_step_general_linear,
                                         host_data_.msc,
                                         host_data_.fluct);

#pragma omp parallel for
    for (size_type i = 0; i < state.size(); ++i)
    {
        CELER_TRY_HANDLE_CONTEXT(
            launch(ThreadId{i}),
            capture_exception,
            KernelContextException(params.ref<MemSpace::host>(),
                                   state.ref(),
                                   ThreadId{i},
                                   this->label()));
    }
    log_and_rethrow(std::move(capture_exception));
}

//---------------------------------------------------------------------------//
/*!
 * Save references from host/device data.
 */
template<MemSpace M>
AlongStepGeneralLinearAction::ExternalRefs<M>::ExternalRefs(
    SPConstFluctuations const& fluct_params, SPConstMsc const& msc_params)
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
}  // namespace celeritas
