//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/model/WavelengthShiftModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/data/CollectionMirror.hh"
#include "celeritas/io/ImportOpticalMaterial.hh"
#include "celeritas/optical/ImportedModelAdapter.hh"
#include "celeritas/optical/Model.hh"

#include "../WavelengthShiftData.hh"

namespace celeritas
{
struct ImportData;

namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Set up and launch the optical WLS model interaction.
 */
class WavelengthShiftModel : public Model
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstImported = std::shared_ptr<ImportedModels const>;
    using HostRef = HostCRef<WavelengthShiftData>;
    using DeviceRef = DeviceCRef<WavelengthShiftData>;
    //!@}

    //! Material-dependent WLS data, indexed by \c OptMatId
    struct Input
    {
        ImportModelClass model{ImportModelClass::size_};
        std::vector<ImportWavelengthShift> data;
        WlsTimeProfile time_profile{WlsTimeProfile::size_};
    };

  public:
    // Create a model builder from imported data
    static ModelBuilder make_builder(SPConstImported, Input);

    // Construct with WLS input data
    WavelengthShiftModel(ActionId, SPConstImported, Input);

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
    ImportedModelAdapter imported_;
    CollectionMirror<WavelengthShiftData> data_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
