//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/model/RayleighModel.cc
//---------------------------------------------------------------------------//
#include "RayleighModel.hh"

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/io/ImportOpticalMaterial.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/ImportedMaterials.hh"
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
 * Create a model builder for Rayleigh scattering from imported data and
 * material parameters.
 */
auto RayleighModel::make_builder(SPConstImported imported,
                                 Input input) -> ModelBuilder
{
    CELER_EXPECT(imported);
    return [imported = std::move(imported),
            input = std::move(input)](ActionId id) {
        return std::make_shared<RayleighModel>(id, imported, input);
    };
}

//---------------------------------------------------------------------------//
/*!
 * Construct the model from imported data and imported material parameters.
 *
 * Uses \c RayleighMfpCalculator to calculate missing imported MFPs from
 * material parameters, if available.
 */
RayleighModel::RayleighModel(ActionId id, SPConstImported imported, Input input)
    : Model(id, "optical-rayleigh", "interact by optical Rayleigh")
    , imported_(ImportModelClass::rayleigh, std::move(imported))
    , input_(std::move(input))
{
    CELER_EXPECT(!input_
                 || input_.materials->num_materials()
                        == imported_.num_materials());

    for (auto mat : range(OptMatId(imported_.num_materials())))
    {
        if (input_)
        {
            CELER_VALIDATE(
                imported_.mfp(mat) || input_.imported_materials->rayleigh(mat),
                << "Rayleigh model requires either imported MFP or "
                   "material parameters to build MFPs for each optical "
                   "material");
        }
        else
        {
            CELER_VALIDATE(imported_.mfp(mat),
                           << "Rayleigh model requires imported MFP for each "
                              "optical material");
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Build the mean free paths for the model.
 */
void RayleighModel::build_mfps(OptMatId mat, MfpBuilder& build) const
{
    CELER_EXPECT(mat < imported_.num_materials());

    if (auto const& mfp = imported_.mfp(mat))
    {
        build(mfp);
    }
    else
    {
        auto mat_view = input_.materials->get(mat);
        auto core_mat_view
            = input_.core_materials->get(mat_view.core_material_id());
        CELER_VALIDATE(core_mat_view.temperature() > 0,
                       << "calculating Rayleigh MFPs from material parameters "
                          "requires positive temperatures");

        RayleighMfpCalculator calc_mfp(
            mat_view, input_.imported_materials->rayleigh(mat), core_mat_view);
        auto energy = calc_mfp.grid().values();

        // Use index of refraction energy grid as calculated MFP energy grid
        inp::Grid grid;
        grid.x = {energy.begin(), energy.end()};
        grid.y.reserve(grid.x.size());
        for (real_type e : grid.x)
        {
            grid.y.push_back(calc_mfp(units::MevEnergy{e}));
        }
        build(grid);
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
