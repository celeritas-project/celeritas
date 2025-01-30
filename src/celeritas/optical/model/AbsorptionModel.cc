//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/model/AbsorptionModel.cc
//---------------------------------------------------------------------------//
#include "AbsorptionModel.hh"

#include "corecel/Assert.hh"
#include "celeritas/optical/action/ActionLauncher.hh"
#include "celeritas/optical/action/TrackSlotExecutor.hh"

#include "AbsorptionExecutor.hh"
#include "../CoreParams.hh"
#include "../CoreState.hh"
#include "../MfpBuilder.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Create a model builder for the optical absorption model.
 */
auto AbsorptionModel::make_builder(SPConstImported imported) -> ModelBuilder
{
    CELER_EXPECT(imported);
    return [i = std::move(imported)](ActionId id) {
        return std::make_shared<AbsorptionModel>(id, i);
    };
}

//---------------------------------------------------------------------------//
/*!
 * Construct the model from imported data.
 */
AbsorptionModel::AbsorptionModel(ActionId id, SPConstImported imported)
    : Model(id, "absorption", "interact by optical absorption")
    , imported_(ImportModelClass::absorption, imported)
{
}

//---------------------------------------------------------------------------//
/*!
 * Build the mean free paths for the model.
 */
void AbsorptionModel::build_mfps(OpticalMaterialId mat, MfpBuilder& build) const
{
    CELER_EXPECT(mat < imported_.num_materials());
    build(imported_.mfp(mat));
}

//---------------------------------------------------------------------------//
/*!
 * Execute the model on the host.
 */
void AbsorptionModel::step(CoreParams const& core_params,
                           CoreStateHost& core_state) const
{
    launch_action(
        core_state,
        make_action_thread_executor(core_params.ptr<MemSpace::native>(),
                                    core_state.ptr(),
                                    this->action_id(),
                                    AbsorptionExecutor{}));
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
