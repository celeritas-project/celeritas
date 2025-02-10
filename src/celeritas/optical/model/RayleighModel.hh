//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/model/RayleighModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "../ImportedModelAdapter.hh"
#include "../Model.hh"
#include "../Types.hh"

namespace celeritas
{
class MaterialParams;
struct ImportOpticalRayleigh;

namespace optical
{
class ImportedMaterials;
class MaterialParams;
//---------------------------------------------------------------------------//
/*!
 * Set up and launch the optical Rayleigh scattering model interaction.
 */
class RayleighModel : public Model
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstImported = std::shared_ptr<ImportedModels const>;
    using SPConstImportedMaterials = std::shared_ptr<ImportedMaterials const>;
    using SPConstMaterials = std::shared_ptr<MaterialParams const>;
    using SPConstCoreMaterials
        = std::shared_ptr<::celeritas::MaterialParams const>;
    //!@}

    //! Optional input for calculating MFP tables from material parameters
    struct Input
    {
        SPConstMaterials materials;
        SPConstCoreMaterials core_materials;
        SPConstImportedMaterials imported_materials;

        //! Whether data is available to calculate material MFP tables
        explicit operator bool() const
        {
            return materials && core_materials && imported_materials;
        }
    };

  public:
    // Create a model builder from imported data and material parameters
    static ModelBuilder make_builder(SPConstImported, Input);

    // Construct with imported data and imported material parameters
    RayleighModel(ActionId id, SPConstImported imported, Input input);

    // Build the mean free paths for this model
    void build_mfps(OpticalMaterialId, MfpBuilder&) const final;

    // Execute the model with host data
    void step(CoreParams const&, CoreStateHost&) const final;

    // Execute the model with device data
    void step(CoreParams const&, CoreStateDevice&) const final;

  private:
    ImportedModelAdapter imported_;
    Input input_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
