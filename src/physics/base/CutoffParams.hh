//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file CutoffParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include "CutoffInterface.hh"
#include "base/PieMirror.hh"
#include "physics/base/Units.hh"
#include <vector>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Data management for particle and material cutoff values.
 */
class CutoffParams
{
  public:
    //!@{
    //! References to constructed data
    using HostRef
        = CutoffParamsData<Ownership::const_reference, MemSpace::host>;
    using DeviceRef
        = CutoffParamsData<Ownership::const_reference, MemSpace::device>;
    //!@}

    //! Input data to construct this class
    using MaterialCutoff = std::vector<SingleCutoff>;
    using Input          = std::vector<MaterialCutoff>;

  public:
    //! Construct with cutoff input data
    explicit CutoffParams(Input& inp);

    //! Access cutoff data on the host
    const HostRef& host_pointers() const { return data_.host(); }

    //! Access cutoff data on the device
    const DeviceRef& device_pointers() const { return data_.device(); }

  private:
    // Host/device storage and reference
    PieMirror<CutoffParamsData> data_;
    using HostValue = CutoffParamsData<Ownership::value, MemSpace::host>;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "CutoffParams.i.hh"
