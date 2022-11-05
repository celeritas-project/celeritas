//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/detail/StepGatherAction.cc
//---------------------------------------------------------------------------//
#include "StepGatherAction.hh"

#include "corecel/Macros.hh"

#include "StepGatherLauncher.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with next action ID.
 */
template<StepPoint P>
StepGatherAction<P>::StepGatherAction(ActionId        id,
                                      SPStepStorage   storage,
                                      SPStepInterface callback)
    : id_(id), callback_(std::move(callback)), storage_(std::move(storage))
{
    CELER_EXPECT(id_);
    CELER_EXPECT(callback_ || P == StepPoint::pre);
    CELER_EXPECT(storage_);
}

//---------------------------------------------------------------------------//
/*!
 * Descriptive name of the action for user output.
 */
template<StepPoint P>
std::string StepGatherAction<P>::description() const
{
    return P == StepPoint::pre    ? "pre-step state collection"
           : P == StepPoint::post ? "post-step state collection"
                                  : "";
}

//---------------------------------------------------------------------------//
/*!
 * Launch the along-step action on host.
 */
template<StepPoint P>
void StepGatherAction<P>::execute(CoreHostRef const& core) const
{
    CELER_EXPECT(core);

    const auto& step_state = this->get_state(core);
    CELER_ASSERT(step_state.size() == core.states.size());
    StepGatherLauncher<P> launch{core, storage_->params.host_ref(), step_state};
#pragma omp parallel for
    for (size_type i = 0; i < core.states.size(); ++i)
    {
        launch(ThreadId{i});
    }
}

#if !CELER_USE_DEVICE
//---------------------------------------------------------------------------//
/*!
 * Launch the along-step action on device.
 */
template<StepPoint P>
void StepGatherAction<P>::execute(CoreDeviceRef const&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

template class StepGatherAction<StepPoint::pre>;
template class StepGatherAction<StepPoint::post>;

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
