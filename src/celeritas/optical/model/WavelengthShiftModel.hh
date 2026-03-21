//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/model/WavelengthShiftModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/Types.hh"
#include "corecel/data/ParamsDataStore.hh"
#include "celeritas/inp/OpticalPhysics.hh"
#include "celeritas/optical/Model.hh"

#include "../WavelengthShiftData.hh"

namespace celeritas
{
namespace optical
{
class MaterialParams;
//---------------------------------------------------------------------------//
/*!
 * Set up and launch the optical WLS model interaction.
 */
class WavelengthShiftModel : public Model
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstMaterials = std::shared_ptr<MaterialParams const>;
    using HostRef = HostCRef<WavelengthShiftData>;
    using DeviceRef = DeviceCRef<WavelengthShiftData>;
    //!@}

  public:
    // Construct with WLS input data
    WavelengthShiftModel(ActionId,
                         inp::OpticalBulkWavelengthShift,
                         SPConstMaterials const&,
                         std::string label);

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
    inp::OpticalBulkWavelengthShift input_;
    ParamsDataStore<WavelengthShiftData> data_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
