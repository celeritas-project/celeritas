//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/alongstep/AlongStepRZMapFieldMscAction.cc
//---------------------------------------------------------------------------//
#include "AlongStepRZMapFieldMscAction.hh"

#include <utility>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/Ref.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/MultiExceptionHandler.hh"
#include "celeritas/em/FluctuationParams.hh"
#include "celeritas/em/UrbanMscParams.hh"
#include "celeritas/field/RZMapFieldInput.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/global/KernelContextException.hh"
#include "celeritas/global/TrackLauncher.hh"

#include "detail/AlongStepRZMapFieldMsc.hh"
namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct the along-step action from input parameters.
 */
std::shared_ptr<AlongStepRZMapFieldMscAction>
AlongStepRZMapFieldMscAction::from_params(ActionId id,
                                          MaterialParams const& materials,
                                          ParticleParams const& particles,
                                          RZMapFieldInput const& field_input,
                                          SPConstMsc const& msc)
{
    SPConstFluctuations fluct
        = std::make_shared<FluctuationParams>(particles, materials);

    return std::make_shared<AlongStepRZMapFieldMscAction>(
        id, std::move(fluct), field_input, msc);
}

//---------------------------------------------------------------------------//
/*!
 * Construct with next action ID and optional energy loss parameters.
 */
AlongStepRZMapFieldMscAction::AlongStepRZMapFieldMscAction(
    ActionId id,
    SPConstFluctuations fluct,
    RZMapFieldInput const& input,
    SPConstMsc msc)
    : id_(id), fluct_(std::move(fluct)), msc_(std::move(msc))
{
    CELER_EXPECT(id_);
    CELER_EXPECT(input);
    CELER_EXPECT(fluct_);
    CELER_EXPECT(msc_);

    field_ = std::make_shared<RZMapFieldParams>(input);
}

//---------------------------------------------------------------------------//
/*!
 * Launch the along-step action on host.
 */
void AlongStepRZMapFieldMscAction::execute(CoreParams const& params,
                                           CoreStateHost& state) const
{
    MultiExceptionHandler capture_exception;

    auto launch
        = make_along_step_track_launcher(*params.ptr<MemSpace::native>(),
                                         *state.ptr(),
                                         this->action_id(),
                                         detail::along_step_mapfield_msc,
                                         msc_->host_ref(),
                                         field_->host_ref(),
                                         fluct_->host_ref());

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
}  // namespace celeritas
