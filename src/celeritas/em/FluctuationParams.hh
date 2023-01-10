//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/FluctuationParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/data/CollectionMirror.hh"
#include "celeritas/em/data/FluctuationData.hh"

namespace celeritas
{
class MaterialParams;
class ParticleParams;

//---------------------------------------------------------------------------//
/*!
 * Manage data for stochastic energy loss of EM particles.
 */
class FluctuationParams
{
  public:
    //!@{
    //! \name Type aliases
    using HostRef = celeritas::HostCRef<FluctuationData>;
    using DeviceRef = celeritas::DeviceCRef<FluctuationData>;
    //!@}

  public:
    // Construct with particle and material data
    FluctuationParams(ParticleParams const& particles,
                      MaterialParams const& materials);

    //! Access physics properties on the host
    HostRef const& host_ref() const { return data_.host(); }

    //! Access physics properties on the device
    DeviceRef const& device_ref() const { return data_.device(); }

  private:
    CollectionMirror<FluctuationData> data_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
