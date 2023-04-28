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
#include "celeritas/field/RZMapField.hh"
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/global/KernelContextException.hh"
#include "celeritas/global/alongstep/AlongStepLauncher.hh"

#include "AlongStepLauncher.hh"
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
                                          SPConstMsc const& msc,
                                          bool eloss_fluctuation)
{
    SPConstFluctuations fluct;
    if (eloss_fluctuation)
    {
        fluct = std::make_shared<FluctuationParams>(particles, materials);
    }

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

    field_ = std::make_shared<RZMapFieldParams>(input);
}

//---------------------------------------------------------------------------//
//! Default destructor
AlongStepRZMapFieldMscAction::~AlongStepRZMapFieldMscAction() = default;

//---------------------------------------------------------------------------//
/*!
 * Launch the along-step action on host.
 */
void AlongStepRZMapFieldMscAction::execute(ParamsHostCRef const& params,
                                           StateHostRef& state) const
{
    CELER_EXPECT(params && state);
    MultiExceptionHandler capture_exception;

    auto launch = make_along_step_launcher(params,
                                           state,
                                           msc_->host_ref(),
                                           field_->host_ref(),
                                           fluct_->host_ref(),
                                           detail::along_step_mapfield_msc);

#pragma omp parallel for
    for (size_type i = 0; i < state.size(); ++i)
    {
        CELER_TRY_HANDLE_CONTEXT(
            launch(ThreadId{i}),
            capture_exception,
            KernelContextException(params, state, ThreadId{i}, this->label()));
    }
    log_and_rethrow(std::move(capture_exception));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
