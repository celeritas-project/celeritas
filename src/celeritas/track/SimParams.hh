//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/SimParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <unordered_map>

#include "corecel/Types.hh"
#include "corecel/data/CollectionMirror.hh"
#include "celeritas/phys/PDGNumber.hh"

#include "SimData.hh"

namespace celeritas
{
class ParticleParams;
struct ImportData;

//---------------------------------------------------------------------------//
/*!
 * Manage persistent simulation data.
 */
class SimParams
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstParticles = std::shared_ptr<ParticleParams const>;
    using HostRef = HostCRef<SimParamsData>;
    using DeviceRef = DeviceCRef<SimParamsData>;
    //!@}

    //! Input data to construct this class
    struct Input
    {
        SPConstParticles particles;
        std::unordered_map<PDGNumber, LoopingThreshold> looping;
    };

  public:
    // Construct with imported data
    static std::shared_ptr<SimParams>
    from_import(ImportData const&, SPConstParticles);

    // Construct with simulation input data
    explicit SimParams(Input const&);

    //! Access data on host
    HostRef const& host_ref() const { return data_.host(); }

    //! Access data on device
    DeviceRef const& device_ref() const { return data_.device(); }

  private:
    // Host/device storage and reference
    CollectionMirror<SimParamsData> data_;
    using HostValue = HostVal<SimParamsData>;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
