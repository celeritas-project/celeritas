//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/model/RayleighModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "celeritas/inp/OpticalPhysics.hh"
#include "celeritas/optical/Model.hh"
#include "celeritas/optical/Types.hh"

namespace celeritas
{
class MaterialParams;

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
    using SPConstMaterials = std::shared_ptr<MaterialParams const>;
    using SPConstCoreMaterials
        = std::shared_ptr<::celeritas::MaterialParams const>;
    //!@}

  public:
    // Construct with imported data and imported material parameters
    RayleighModel(ActionId,
                  inp::OpticalBulkRayleigh,
                  SPConstMaterials const&,
                  SPConstCoreMaterials const&);

    // Build the mean free paths for this model
    void build_mfps(OptMatId, MfpBuilder&) const final;

    // Execute the model with host data
    void step(CoreParams const&, CoreStateHost&) const final;

    // Execute the model with device data
    void step(CoreParams const&, CoreStateDevice&) const final;

  private:
    inp::OpticalBulkRayleigh input_;
    SPConstMaterials materials_;
    SPConstCoreMaterials core_materials_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
