//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/model/AbsorptionModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/inp/OpticalPhysics.hh"
#include "celeritas/optical/Model.hh"
#include "celeritas/optical/Types.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Set up and launch the optical absorption model interaction.
 */
class AbsorptionModel final : public Model
{
  public:
    // Construct with input data
    AbsorptionModel(ActionId id, inp::OpticalBulkAbsorption const& input);

    // Build the mean free paths for this model
    void build_mfps(OptMatId mat, MfpBuilder&) const final;

    // Execute the model with host data
    void step(CoreParams const&, CoreStateHost&) const final;

    // Execute the model with device data
    void step(CoreParams const&, CoreStateDevice&) const final;

  private:
    inp::OpticalBulkAbsorption const& input_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
