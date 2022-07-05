//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/FluctuationParams.hh
//---------------------------------------------------------------------------//
#pragma once

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
    using SPConstParticles = std::shared_ptr<const ParticleParams>;
    using SPConstMaterials = std::shared_ptr<const MaterialParams>;

    using HostRef   = ParamsHostRef<FluctuationData>;
    using DeviceRef = ParamsDeviceRef<FluctuationData>;
    //!@}

  public:
    // Construct with particle and material data
    FluctuationParams(SPConstParticles particles, SPConstMaterials materials);

    //! Access physics properties on the host
    const HostRef& host_ref() const { return data_.host(); }

    //! Access physics properties on the device
    const DeviceRef& device_ref() const { return data_.device(); }

  private:
    CollectionMirror<FluctuationData> data_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
