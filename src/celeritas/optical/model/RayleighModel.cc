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
#include "celeritas/optical/MaterialParams.hh"
#include "celeritas/optical/MfpBuilder.hh"

#include "RayleighMfpCalculator.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Construct the model from imported data.
 */
RayleighModel::RayleighModel(
    ActionId id,
    SPConstImported imported,
    SPConstMaterials materials,
    std::vector<OpticalRayleighMaterial> rayleigh_materials)
    : Model(id, "optical-rayleigh", "interact by optical Rayleigh")
    , imported_(ImportModelClass::rayleigh, imported)
    , materials_(std::move(materials))
    , rayleigh_materials_(std::move(rayleigh_materials))
{
    CELER_EXPECT(materials_);
    CELER_EXPECT(materials_->num_materials() == imported_.num_materials());
    CELER_EXPECT(materials_->num_materials() == rayleigh_materials_.size());

    for (auto mat : range(OpticalMaterialId(materials_->num_materials())))
    {
        CELER_EXPECT(imported_.mfp(mat) || rayleigh_materials_[mat.get()]);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Build the mean free paths for the model.
 */
void RayleighModel::build_mfps(OpticalMaterialId mat, MfpBuilder& build) const
{
    CELER_EXPECT(mat < materials_->num_materials());

    if (auto const& mfp = imported_.mfp(mat))
    {
        build(mfp);
    }
    else
    {
        RayleighMfpCalculator calc_mfp(
            MaterialView(materials_->host_ref(), mat),
            rayleigh_materials_[mat.get()]);

        // Use index of refraction energy grid as calculated MFP energy grid
        auto const& energy_grid = calc_mfp.grid();

        ImportPhysicsVector result{ImportPhysicsVectorType::free,
                                   std::vector<double>(energy_grid.size()),
                                   std::vector<double>(energy_grid.size())};

        for (auto i : range(energy_grid.size()))
        {
            result.x[i] = energy_grid[i];
            result.y[i] = calc_mfp(celeritas::units::MevEnergy{result.x[i]});
        }

        build(result);
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
