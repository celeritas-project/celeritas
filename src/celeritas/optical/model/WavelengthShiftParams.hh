//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/model/WavelengthShiftParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "celeritas/io/ImportOpticalMaterial.hh"

#include "../WavelengthShiftData.hh"

namespace celeritas
{
struct ImportData;

namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Build and manage wavelength shift (WLS) data.
 *
 * \todo Replace with \c WavelengthShiftModel
 */
class WavelengthShiftParams final
    : public ParamsDataInterface<WavelengthShiftData>
{
  public:
    //! Material-dependent WLS data, indexed by \c OptMatId
    struct Input
    {
        std::vector<ImportWavelengthShift> data;
    };

  public:
    // Construct with imported data
    static std::shared_ptr<WavelengthShiftParams>
    from_import(ImportData const& data);

    // Construct with WLS input data
    explicit WavelengthShiftParams(Input const& input);

    //! Access WLS data on the host
    HostRef const& host_ref() const final { return data_.host_ref(); }

    //! Access WLS data on the device
    DeviceRef const& device_ref() const final { return data_.device_ref(); }

  private:
    // Host/device storage and reference
    CollectionMirror<WavelengthShiftData> data_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
