//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/model/AbsorptionModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/optical/ImportMaterialAdapter.hh"
#include "celeritas/optical/Model.hh"

namespace celeritas
{
namespace optical
{

//---------------------------------------------------------------------------//
/*!
 * Set up and launch the optical absorption model interaction.
 */
class AbsorptionModel : public Model
{
  public:
    using SPConstImported = std::shared_ptr<ImportedMaterials const>;
    using ImportedAdapter = ImportMaterialAdapter<ImportModelClass::absorption>;

  public:
    // Construct with imported data
    AbsorptionModel(ActionId id, SPConstImported imported);

    // Build the mean free paths for this model
    void build_mfp(OpticalMaterialId, detail::MfpBuilder) const override final;

    // Execute the model with host data
    void step(CoreParams const&, CoreStateHost&) const override final;

    // Execute the model with device data
    void step(CoreParams const&, CoreStateDevice&) const override final;

  private:
    ImportedAdapter imported_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
