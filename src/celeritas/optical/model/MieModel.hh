//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/model/MieModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/data/ParamsDataStore.hh"
#include "celeritas/optical/ImportedModelAdapter.hh"
#include "celeritas/optical/Model.hh"

#include "../MieData.hh"

namespace celeritas
{
struct ImportData;
struct ImportMie;
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Set up and launch the optical Mie scattering model interaction
 */
class MieModel final : public Model
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstImported = std::shared_ptr<ImportedModels const>;
    using HostRef = HostCRef<MieData>;
    using DeviceRef = DeviceCRef<MieData>;
    //!@}

    //! Material-dependent mie scattering parameter data, indexed by \c
    //! OptMatId
    struct Input
    {
        ImportModelClass model{ImportModelClass::size_};
        std::vector<ImportMie> data;
    };

  public:
    // Create a model builder from imported data
    static ModelBuilder make_builder(SPConstImported imported, Input input);

    // Construct with mie scattering input data
    MieModel(ActionId id, SPConstImported imported, Input input);

    // Build the mean free paths for this model
    void build_mfps(OptMatId mat, MfpBuilder& build) const final;

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
    ParamsDataStore<MieData> data_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
