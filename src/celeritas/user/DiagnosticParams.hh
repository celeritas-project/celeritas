//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/DiagnosticParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"

#include "DiagnosticData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Manage persistent data for user diagnostics.
 */
class DiagnosticParams final : public ParamsDataInterface<DiagnosticParamsData>
{
  public:
    //! Construction arguments
    struct Input
    {
        bool field_diagnostic{false};  //!< Whether field diagnostic is enabled
    };

  public:
    // Construct with list of enabled diagnostics
    explicit DiagnosticParams(Input const&);

    //! Access data on host
    HostRef const& host_ref() const final { return data_.host(); }

    //! Access data on device
    DeviceRef const& device_ref() const final { return data_.device(); }

  private:
    // Host/device storage and reference
    CollectionMirror<DiagnosticParamsData> data_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
