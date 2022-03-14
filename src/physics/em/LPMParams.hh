//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LPMParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/CollectionMirror.hh"
#include "base/Types.hh"

#include "LPMData.hh"

namespace celeritas
{
class ParticleParams;

//---------------------------------------------------------------------------//
/*!
 * Data management for the LPM table data used in relativistic models.
 */
class LPMParams
{
  public:
    //@{
    //! Type aliases
    using HostRef   = LPMData<Ownership::const_reference, MemSpace::host>;
    using DeviceRef = LPMData<Ownership::const_reference, MemSpace::device>;
    using SPConstParticles = std::shared_ptr<const ParticleParams>;
    //@}

  public:
    // Construct from shared particle and material data
    explicit LPMParams(SPConstParticles particles);

    // Access LPM data on the host
    const HostRef& host_ref() const { return data_.host(); }

    // Access LPM data on the device
    const DeviceRef& device_ref() const { return data_.device(); }

  private:
    //// DATA ////

    // Host/device storage and reference
    CollectionMirror<LPMData> data_;

    //// HELPER FUNCTIONS ////

    MigdalData compute_lpm_data(real_type);

    using HostValue = LPMData<Ownership::value, MemSpace::host>;
};
//---------------------------------------------------------------------------//
} // namespace celeritas
