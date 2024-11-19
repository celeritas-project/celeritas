//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/model/AbsorptionModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "../ImportedModels.hh"
#include "../Model.hh"

namespace celeritas
{
namespace optical
{
struct ModelBuilder;
//---------------------------------------------------------------------------//
/*!
 * Set up and launch the optical absorption model interaction.
 */
class AbsorptionModel final : public Model
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstImported = std::shared_ptr<ImportedModels const>;
    //!@}

  public:
    // Create a model builder for absorption
    static std::shared_ptr<ModelBuilder> make_builder(SPConstImported);

    // Construct with imported data
    AbsorptionModel(ActionId id, SPConstImported imported);

    // Build the mean free paths for this model
    void build_mfps(OpticalMaterialId mat, MfpBuilder&) const final;

    // Execute the model with host data
    void step(CoreParams const&, CoreStateHost&) const final;

    // Execute the model with device data
    void step(CoreParams const&, CoreStateDevice&) const final;

  private:
    ImportedModelAdapter imported_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
