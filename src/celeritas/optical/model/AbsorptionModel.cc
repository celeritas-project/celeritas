//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/model/AbsorptionModel.cc
//---------------------------------------------------------------------------//
#include "AbsorptionModel.hh"

#include "corecel/Assert.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/InteractionApplier.hh"
#include "celeritas/optical/MfpBuilder.hh"
#include "celeritas/optical/action/ActionLauncher.hh"
#include "celeritas/optical/action/TrackSlotExecutor.hh"

#include "AbsorptionExecutor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Construct the model from input data.
 */
AbsorptionModel::AbsorptionModel(ActionId id, inp::OpticalBulkAbsorption input)
    : Model(id, "absorption", "interact by optical absorption")
    , input_(std::move(input))
{
    CELER_VALIDATE(input_, << "invalid input for optical absorption model");
}

//---------------------------------------------------------------------------//
/*!
 * Build the mean free paths for the model.
 */
void AbsorptionModel::build_mfps(OptMatId mat, MfpBuilder& build) const
{
    if (auto iter = input_.materials.find(mat); iter != input_.materials.end())
    {
        build(iter->second.mfp);
    }
    else
    {
        build();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Execute the model on the host.
 */
void AbsorptionModel::step(CoreParams const& params, CoreStateHost& state) const
{
    launch_action(
        state,
        make_action_thread_executor(params.ptr<MemSpace::native>(),
                                    state.ptr(),
                                    this->action_id(),
                                    InteractionApplier{AbsorptionExecutor{}}));
}

//---------------------------------------------------------------------------//
/*!
 * Execute the model on the device.
 */
#if !CELER_USE_DEVICE
void AbsorptionModel::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
