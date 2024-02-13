//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalPropertyParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/Types.hh"
#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "celeritas/io/ImportOpticalMaterial.hh"

#include "OpticalPropertyData.hh"

namespace celeritas
{
struct ImportData;

//---------------------------------------------------------------------------//
/*!
 * Build and manage optical property data for materials.
 */
class OpticalPropertyParams final
    : public ParamsDataInterface<OpticalPropertyData>
{
  public:
    // Shared optical properties, indexed by \c OpticalMaterialId
    struct Input
    {
        std::vector<ImportOpticalProperty> data;
    };

  public:
    // Construct with imported data
    static std::shared_ptr<OpticalPropertyParams>
    from_import(ImportData const& data);

    // Construct with optical property data
    explicit OpticalPropertyParams(Input const& inp);

    //! Access optical properties on the host
    HostRef const& host_ref() const final { return data_.host_ref(); }

    //! Access optical properties on the device
    DeviceRef const& device_ref() const final { return data_.device_ref(); }

  private:
    CollectionMirror<OpticalPropertyData> data_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
