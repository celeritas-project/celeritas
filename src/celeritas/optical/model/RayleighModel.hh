//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/model/RayleighModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "../ImportedModelAdapter.hh"
#include "../Model.hh"

namespace celeritas
{
namespace optical
{
class MaterialParams;
class OpticalRayleighMaterial;

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
    //!@}

  public:
    // Construct with imported data
    RayleighModel(ActionId id,
                  SPConstImported imported,
                  SPConstMaterials materials,
                  std::vector<OpticalRayleighMaterial> rayleigh_materials);

    // Build the mean free paths for this model
    void build_mfps(OpticalMaterialId, MfpBuilder&) const final;

    // Execute the model with host data
    void step(CoreParams const&, CoreStateHost&) const final;

    // Execute the model with device data
    void step(CoreParams const&, CoreStateDevice&) const final;

  private:
    ImportedModelAdapter imported_;
    SPConstMaterials materials_;
    std::vector<OpticalRayleighMaterial> rayleigh_materials_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
