//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/params/FluctuationParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "corecel/data/ParamsDataStore.hh"
#include "celeritas/em/data/FluctuationData.hh"

namespace celeritas
{
class MaterialParams;
class ParticleParams;

//---------------------------------------------------------------------------//
/*!
 * Manage data for stochastic energy loss of EM particles.
 */
class FluctuationParams final : public ParamsDataInterface<FluctuationData>
{
  public:
    // Construct with particle and material data
    FluctuationParams(ParticleParams const& particles,
                      MaterialParams const& materials);

    //! Access physics properties on the host
    HostRef const& host_ref() const final { return data_.host_ref(); }

    //! Access physics properties on the device
    DeviceRef const& device_ref() const final { return data_.device_ref(); }

  private:
    ParamsDataStore<FluctuationData> data_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
