//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/model/OpticalRayleighModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "celeritas/optical/OpticalModel.hh"

#include "OpticalRayleighData.hh"

namespace celeritas
{
class ImportedOpticalMaterials;
//---------------------------------------------------------------------------//
/*!
 * Set up and launch the optical Rayleigh scattering model interaction.
 */
class OpticalRayleighModel : public OpticalModel,
                             public ParamsDataInterface<OpticalRayleighData>
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstImported = std::shared_ptr<ImportedOpticalMaterials const>;
    //!@}

  public:
    //! Construct the model from imported data
    OpticalRayleighModel(ActionId id, SPConstImported imported);

    //! Build mean free paths
    void build_mfp(OpticalModelMfpBuilder& builder) const override final;

    //! Execute model on host
    void execute(OpticalParams const&, OpticalStateHost&) const override final;

    //! Execute model on device
    void
    execute(OpticalParams const&, OpticalStateDevice&) const override final;

    //! Retrieve host reference to model data
    HostRef const& host_ref() const { return data_.host_ref(); }

    //! Retrieve device reference to model data
    DeviceRef const& device_ref() const { return data_.device_ref(); }

  private:
    CollectionMirror<OpticalRayleighData> data_;
    SPConstImported imported_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
