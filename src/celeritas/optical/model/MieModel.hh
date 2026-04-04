//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/model/MieModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "corecel/data/ParamsDataStore.hh"
#include "celeritas/inp/OpticalPhysics.hh"
#include "celeritas/optical/Model.hh"

#include "../MieData.hh"

namespace celeritas
{
namespace optical
{
class MaterialParams;
//---------------------------------------------------------------------------//
/*!
 * Set up and launch the optical Mie scattering model interaction
 */
class MieModel final : public Model
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstMaterials = std::shared_ptr<MaterialParams const>;
    using HostRef = HostCRef<MieData>;
    using DeviceRef = DeviceCRef<MieData>;
    //!@}

  public:
    // Construct with mie scattering input data
    MieModel(ActionId, inp::OpticalBulkMie, SPConstMaterials const&);

    // Build the mean free paths for this model
    void build_mfps(OptMatId, MfpBuilder&) const final;

    // Execute the model with host data
    void step(CoreParams const&, CoreStateHost&) const final;

    // Execute the model with device data
    void step(CoreParams const&, CoreStateDevice&) const final;

    //! Access data on the host
    HostRef const& host_ref() const { return data_.host_ref(); }

    //! Access data on the device
    DeviceRef const& device_ref() const { return data_.device_ref(); }

  private:
    inp::OpticalBulkMie input_;
    ParamsDataStore<MieData> data_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
