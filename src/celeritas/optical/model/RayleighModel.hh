//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/model/RayleighModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "../ImportedModelAdapter.hh"
#include "../Model.hh"

namespace celeritas
{
class MaterialParams;
struct ImportOpticalRayleigh;

namespace optical
{
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
    using SPConstMaterials = std::shared_ptr<MaterialParams const>;
    using SPConstCoreMaterials
        = std::shared_ptr<::celeritas::MaterialParams const>;
    //!@}

  public:
    // Construct with imported data
    RayleighModel(ActionId id, SPConstImported imported);

    // Construct with imported data and imported material parameters
    RayleighModel(ActionId id,
                  SPConstImported imported,
                  SPConstMaterials materials,
                  SPConstCoreMaterials core_materials,
                  std::vector<ImportOpticalRayleigh> rayleigh);

    // Build the mean free paths for this model
    void build_mfps(OpticalMaterialId, MfpBuilder&) const final;

    // Execute the model with host data
    void step(CoreParams const&, CoreStateHost&) const final;

    // Execute the model with device data
    void step(CoreParams const&, CoreStateDevice&) const final;

  private:
    ImportedModelAdapter imported_;
    SPConstMaterials materials_;
    SPConstCoreMaterials core_materials_;
    std::vector<ImportOpticalRayleigh> import_rayleigh_materials_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
