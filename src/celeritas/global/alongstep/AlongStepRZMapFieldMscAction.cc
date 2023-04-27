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
 * Construct with next action ID and optional energy loss parameters.
 */
AlongStepRZMapFieldMscAction::AlongStepRZMapFieldMscAction(
    ActionId id, RZMapFieldInput const& input, SPConstMsc msc)
    : id_(id), msc_(std::move(msc)), host_data_(msc_), device_data_(msc_)
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
                                           host_data_.msc,
                                           field_->host_ref(),
                                           NoData{},
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
/*!
 * Save references from host/device data.
 */
template<MemSpace M>
AlongStepRZMapFieldMscAction::ExternalRefs<M>::ExternalRefs(
    SPConstMsc const& msc_params)
{
    if (M == MemSpace::device && !celeritas::device())
    {
        // Skip device copy if disabled
        return;
    }

    if (msc_params)
    {
        msc = get_ref<M>(*msc_params);
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
