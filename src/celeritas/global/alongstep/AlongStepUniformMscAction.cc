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
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/global/KernelContextException.hh"
#include "celeritas/global/alongstep/AlongStepLauncher.hh"

#include "AlongStepLauncher.hh"
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
void AlongStepUniformMscAction::execute(CoreHostRef const& data) const
{
    CELER_EXPECT(data);

    MultiExceptionHandler capture_exception;
    auto launch = make_along_step_launcher(data,
                                           host_data_.msc,
                                           field_params_,
                                           NoData{},
                                           detail::along_step_uniform_msc);

#pragma omp parallel for
    for (size_type i = 0; i < data.states.size(); ++i)
    {
        CELER_TRY_HANDLE_CONTEXT(
            launch(TrackSlotId{i}),
            capture_exception,
            KernelContextException(data, TrackSlotId{i}, this->label()));
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
