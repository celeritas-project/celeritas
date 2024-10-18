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
struct ImportOpticalRayleigh;
struct ImportOpticalProperty;

namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Set up and launch the optical Rayleigh scattering model interaction.
 */
class RayleighModel : public Model
{
  public:
    struct Input
    {
        std::vector<ImportOpticalProperty> properties;
        std::vector<ImportOpticalRayleigh> rayleigh;
    };

  public:
    // Construct with imported data
    RayleighModel(ActionId id, ImportedModelAdapter imported, Input input);

    // Build the mean free paths for this model
    void build_mfps(OpticalMaterialId, MfpBuilder&) const override final;

    // Execute the model with host data
    void step(CoreParams const&, CoreStateHost&) const override final;

    // Execute the model with device data
    void step(CoreParams const&, CoreStateDevice&) const override final;

  private:
    ImportedModelAdapter imported_;
    std::vector<ImportOpticalProperty> properties_;
    std::vector<ImportOpticalRayleigh> rayleigh_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
