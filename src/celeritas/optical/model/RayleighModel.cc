//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/model/RayleighModel.cc
//---------------------------------------------------------------------------//
#include "RayleighModel.hh"

#include "corecel/Assert.hh"
#include "corecel/cont/VariantUtils.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/io/ImportOpticalMaterial.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/InteractionApplier.hh"
#include "celeritas/optical/MaterialParams.hh"
#include "celeritas/optical/MfpBuilder.hh"
#include "celeritas/optical/action/ActionLauncher.hh"
#include "celeritas/optical/action/TrackSlotExecutor.hh"

#include "RayleighExecutor.hh"
#include "RayleighMfpCalculator.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Construct the model from imported data and imported material parameters.
 *
 * Uses \c RayleighMfpCalculator to calculate missing imported MFPs from
 * material parameters, if available.
 */
RayleighModel::RayleighModel(ActionId id,
                             inp::OpticalBulkRayleigh input,
                             SPConstMaterials const& materials,
                             SPConstCoreMaterials const& core_materials)
    : Model(id, "optical-rayleigh", "interact by optical Rayleigh")
    , input_(std::move(input))
    , materials_(materials)
    , core_materials_(core_materials)
{
    CELER_EXPECT(materials_);
    CELER_EXPECT(core_materials_);

    CELER_VALIDATE(input_, << "invalid input for optical Rayleigh model");
}

//---------------------------------------------------------------------------//
/*!
 * Build the mean free paths for the model.
 */
void RayleighModel::build_mfps(OptMatId mat, MfpBuilder& build) const
{
    CELER_EXPECT(mat < materials_->num_materials());

    if (auto iter = input_.materials.find(mat); iter != input_.materials.end())
    {
        CELER_ASSERT(iter->second);
        std::visit(
            (Overload{[&](inp::Grid const& grid) { build(grid); },
                      [&](inp::OpticalRayleighAnalytic const& analytic) {
                          // MFPs can be calculated from user given propcerties
                          auto mat_view = materials_->get(mat);
                          auto core_mat_view = core_materials_->get(
                              mat_view.core_material_id());
                          CELER_VALIDATE(core_mat_view.temperature() > 0,
                                         << "calculating Rayleigh MFPs from "
                                            "material parameters requires "
                                            "positive temperatures");

                          RayleighMfpCalculator calc_mfp(
                              mat_view, analytic, core_mat_view);
                          auto energy = calc_mfp.grid().values();

                          // Use refractive index energy grid for MFP
                          inp::Grid grid;
                          grid.x = {energy.begin(), energy.end()};
                          grid.y.reserve(grid.x.size());
                          for (real_type e : grid.x)
                          {
                              grid.y.push_back(calc_mfp(units::MevEnergy{e}));
                          }
                          build(grid);
                      }}),
            iter->second.mfp);
    }
    else
    {
        // Build a grid with infinite MFP that prevents selection of the model
        build();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Execute the model on the host.
 */
void RayleighModel::step(CoreParams const& params, CoreStateHost& state) const
{
    launch_action(
        state,
        make_action_thread_executor(params.ptr<MemSpace::native>(),
                                    state.ptr(),
                                    this->action_id(),
                                    InteractionApplier{RayleighExecutor{}}));
}

//---------------------------------------------------------------------------//
/*!
 * Execute the model on the device.
 */
#if !CELER_USE_DEVICE
void RayleighModel::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
