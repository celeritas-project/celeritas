//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/AbsorptionModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "celeritas/optical/OpticalModel.hh"

namespace celeritas
{
class ImportOpticalMaterial;
class ImportOpticalAbsorption;

//---------------------------------------------------------------------------//
/*!
 * Set up and launch the optical absorption model interaction.
 */
class AbsorptionModel : public OpticalModel
{
  public:
    //! Construct model with imported data
    AbsorptionModel(ActionId id,
                    std::vector<ImportOpticalMaterial> const& imported);

    //! Build the mean free paths for this model
    void build_mfp(OpticalModelMfpBuilder& build) const override final;

    //! Execute the model with host data
    void execute(OpticalParams const&, OpticalStateHost&) const override final;

    //! Execute the model with device data
    void execute(OpticalParams const&, OpticalStateDevice&) const override final;

  private:
    std::vector<ImportOpticalAbsorption> imported_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
