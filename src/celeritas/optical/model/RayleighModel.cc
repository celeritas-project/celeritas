//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/model/RayleighModel.cc
//---------------------------------------------------------------------------//
#include "RayleighModel.hh"

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/io/ImportOpticalMaterial.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/optical/MaterialParams.hh"
#include "celeritas/optical/MfpBuilder.hh"

#include "RayleighMfpCalculator.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Search for material ID corresponding to the given optical material ID.
 *
 * Assumes material ID to optical material ID mapping is one-to-one.
 */
::celeritas::MaterialId
get_core_material(::celeritas::MaterialParams const& params,
                  OpticalMaterialId opt_mat)
{
    for (auto mat : range(::celeritas::MaterialId{params.num_materials()}))
    {
        if (::celeritas::MaterialView(params.host_ref(), mat)
                .optical_material_id()
            == opt_mat)
        {
            return mat;
        }
    }
    return ::celeritas::MaterialId{};
}

//---------------------------------------------------------------------------//
/*!
 * Construct the model from imported data.
 */
RayleighModel::RayleighModel(ActionId id, SPConstImported imported)
    : Model(id, "optical-rayleigh", "interact by optical Rayleigh")
    , imported_(ImportModelClass::rayleigh, imported)
{
    for (auto mat : range(OpticalMaterialId(imported_.num_materials())))
    {
        CELER_VALIDATE(imported_.mfp(mat),
                       << "Rayleigh model requires imported MFP for each "
                          "optical material");
    }
}

//---------------------------------------------------------------------------//
/*!
 * Construct the model from imported data and imported material parameters.
 *
 * Uses \c RayleighMfpCalculator to calculate missing imported MFPs from
 * material parameters.
 */
RayleighModel::RayleighModel(
    ActionId id,
    SPConstImported imported,
    SPConstMaterials materials,
    SPConstCoreMaterials core_materials,
    std::vector<ImportOpticalRayleigh> import_rayleigh_materials)
    : Model(id, "optical-rayleigh", "interact by optical Rayleigh")
    , imported_(ImportModelClass::rayleigh, imported)
    , materials_(std::move(materials))
    , core_materials_(std::move(core_materials))
    , import_rayleigh_materials_(std::move(import_rayleigh_materials))
{
    CELER_EXPECT(materials_);
    CELER_EXPECT(core_materials_);
    CELER_EXPECT(materials_->num_materials() == imported_.num_materials());
    CELER_EXPECT(materials_->num_materials()
                 == import_rayleigh_materials_.size());

    for (auto mat : range(OpticalMaterialId(materials_->num_materials())))
    {
        CELER_VALIDATE(imported_.mfp(mat)
                           || import_rayleigh_materials_[mat.get()],
                       << "Rayleigh model requires either imported MFP or "
                          "material parameters to build MFPs for each optical "
                          "material");
    }
}

//---------------------------------------------------------------------------//
/*!
 * Build the mean free paths for the model.
 */
void RayleighModel::build_mfps(OpticalMaterialId mat, MfpBuilder& build) const
{
    CELER_EXPECT(mat < imported_.num_materials());

    if (auto const& mfp = imported_.mfp(mat))
    {
        build(mfp);
    }
    else
    {
        ::celeritas::MaterialId core_mat
            = get_core_material(*core_materials_, mat);
        CELER_VALIDATE(core_mat,
                       << "calculating Rayleigh MFPs from material parameters "
                          "requires core material corresponding to each "
                          "optical material");

        ::celeritas::MaterialView core_mat_view(core_materials_->host_ref(),
                                                core_mat);
        CELER_VALIDATE(core_mat_view.temperature() > 0,
                       << "calculating Rayleigh MFPs from material parameters "
                          "requires positive temperatures");

        RayleighMfpCalculator calc_mfp(
            MaterialView(materials_->host_ref(), mat),
            import_rayleigh_materials_[mat.get()],
            core_mat_view);

        // Use index of refraction energy grid as calculated MFP energy grid
        auto const& energy_grid = calc_mfp.grid().values();

        std::vector<real_type> mfp_grid;
        mfp_grid.reserve(energy_grid.size());
        for (real_type energy : energy_grid)
        {
            mfp_grid.push_back(calc_mfp(celeritas::units::MevEnergy{energy}));
        }

        build(energy_grid, make_span(mfp_grid));
    }
}

//---------------------------------------------------------------------------//
/*!
 * Execute the model on the host.
 */
void RayleighModel::step(CoreParams const&, CoreStateHost&) const
{
    CELER_NOT_IMPLEMENTED("optical core physics");
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
