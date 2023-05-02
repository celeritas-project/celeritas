//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/alongstep/AlongStepUniformMscAction.cc
//---------------------------------------------------------------------------//
#include "AlongStepUniformMscAction.hh"

#include <utility>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/Ref.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/MultiExceptionHandler.hh"
#include "celeritas/em/UrbanMscParams.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/global/KernelContextException.hh"
#include "celeritas/global/TrackLauncher.hh"

#include "detail/AlongStepUniformMsc.hh"
namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with next action ID and optional energy loss parameters.
 */
AlongStepUniformMscAction::AlongStepUniformMscAction(
    ActionId id, UniformFieldParams const& field_params, SPConstMsc msc)
    : id_(id)
    , msc_(std::move(msc))
    , field_params_(field_params)
    , host_data_(msc_)
    , device_data_(msc_)
{
    CELER_EXPECT(id_);
}

//---------------------------------------------------------------------------//
//! Default destructor
AlongStepUniformMscAction::~AlongStepUniformMscAction() = default;

//---------------------------------------------------------------------------//
/*!
 * Launch the along-step action on host.
 */
void AlongStepUniformMscAction::execute(CoreParams const& params,
                                        CoreStateHost& state) const
{
    MultiExceptionHandler capture_exception;
    auto launch = make_active_track_launcher(params.ref<MemSpace::native>(),
                                             state.ref(),
                                             detail::along_step_uniform_msc,
                                             host_data_.msc,
                                             field_params_);

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
AlongStepUniformMscAction::ExternalRefs<M>::ExternalRefs(
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
