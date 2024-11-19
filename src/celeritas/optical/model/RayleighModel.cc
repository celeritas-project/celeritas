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

#include "RayleighMfpCalculator.hh"
#include "../ImportedMaterials.hh"
#include "../MaterialParams.hh"
#include "../MfpBuilder.hh"
#include "../ModelBuilder.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Builder for optical Rayleigh scattering model.
 */
class RayleighModelBuilder final : public ModelBuilder
{
  public:
    //!@{
    //! \name Type aliases
    using SPModel = ModelBuilder::SPModel;
    using SPConstImported = RayleighModel::SPConstImported;
    using Input = RayleighModel::Input;
    //!@}

  public:
    RayleighModelBuilder(SPConstImported imported, Input input)
        : imported_(std::move(imported)), input_(std::move(input))
    {
        CELER_EXPECT(imported_);
    }

    SPModel operator()(ActionId id) const final
    {
        return std::make_shared<RayleighModel>(id, imported_, input_);
    }

  private:
    SPConstImported imported_;
    Input input_;
};

//---------------------------------------------------------------------------//
/*!
 * Create a model builder for Rayleigh scattering from imported data and
 * material parameters.
 */
std::shared_ptr<ModelBuilder>
RayleighModel::make_builder(SPConstImported imported, Input input)
{
    return std::make_shared<RayleighModelBuilder>(std::move(imported),
                                                  std::move(input));
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
    if (input)
    {
        CELER_EXPECT(input_.materials->num_materials()
                     == imported_.num_materials());

        for (auto mat :
             range(OpticalMaterialId(input_.materials->num_materials())))
        {
            CELER_VALIDATE(
                imported_.mfp(mat) || input_.imported_materials->rayleigh(mat),
                << "Rayleigh model requires either imported MFP or "
                   "material parameters to build MFPs for each optical "
                   "material");
        }
    }
    else
    {
        for (auto mat : range(OpticalMaterialId(imported_.num_materials())))
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
void RayleighModel::build_mfps(OpticalMaterialId mat, MfpBuilder& build) const
{
    CELER_EXPECT(mat < imported_.num_materials());

    if (auto const& mfp = imported_.mfp(mat))
    {
        build(mfp);
    }
    else
    {
        auto core_mat_view = input_.core_materials->get(
            input_.imported_materials->core_material_id(mat));
        CELER_VALIDATE(core_mat_view.temperature() > 0,
                       << "calculating Rayleigh MFPs from material parameters "
                          "requires positive temperatures");

        RayleighMfpCalculator calc_mfp(input_.materials->get(mat),
                                       input_.imported_materials->rayleigh(mat),
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
