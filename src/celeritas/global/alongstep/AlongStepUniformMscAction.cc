//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/alongstep/AlongStepUniformMscAction.cc
//---------------------------------------------------------------------------//
#include "AlongStepUniformMscAction.hh"

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/data/Ref.hh"
#include "corecel/sys/MultiExceptionHandler.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/em/model/UrbanMscModel.hh"
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/global/alongstep/detail/AlongStepLauncherImpl.hh"
#include "celeritas/phys/PhysicsParams.hh"

#include "AlongStepLauncher.hh"
#include "detail/AlongStepUniformMsc.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct the along-step action from input parameters.
 */
std::shared_ptr<AlongStepUniformMscAction>
AlongStepUniformMscAction::from_params(ActionId                  id,
                                       const PhysicsParams&      physics,
                                       const UniformFieldParams& field_params)
{
    // TODO: Super hacky!! This will be cleaned up later.
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

    return std::make_shared<AlongStepUniformMscAction>(
        id, field_params, std::move(msc));
}

//---------------------------------------------------------------------------//
/*!
 * Construct with next action ID and optional energy loss parameters.
 */
AlongStepUniformMscAction::AlongStepUniformMscAction(
    ActionId id, const UniformFieldParams& field_params, SPConstMsc msc)
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
        CELER_TRY_ELSE(launch(ThreadId{i}), capture_exception);
    }
    log_and_rethrow(std::move(capture_exception));
}

//---------------------------------------------------------------------------//
/*!
 * Save references from host/device data.
 */
template<MemSpace M>
AlongStepUniformMscAction::ExternalRefs<M>::ExternalRefs(
    const SPConstMsc& msc_params)
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
} // namespace celeritas
