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
struct MaterialParams;

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
    //!@}

    struct Input
    {
        std::vector<ImportOpticalProperty> properties;
        std::vector<ImportOpticalRayleigh> rayleigh;
    };

  public:
    // Construct with imported data
    RayleighModel(ActionId id, SPConstImported imported, Input input);

    // Build the mean free paths for this model
    void build_mfps(OpticalMaterialId, MfpBuilder&) const final;

    // Execute the model with host data
    void step(CoreParams const&, CoreStateHost&) const final;

    // Execute the model with device data
    void step(CoreParams const&, CoreStateDevice&) const final;

  private:
    ImportedModelAdapter imported_;
    std::vector<ImportOpticalProperty> properties_;
    std::vector<ImportOpticalRayleigh> rayleigh_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
