//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/ScintillationParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "corecel/data/ParamsDataStore.hh"
#include "celeritas/inp/OpticalPhysics.hh"

#include "ScintillationData.hh"

namespace celeritas
{
namespace optical
{
class MaterialParams;
}

//---------------------------------------------------------------------------//
/*!
 * Build and manage scintillation data.
 *
 * Celeritas contains special Gaussian distributions for scintillation that
 * aren't supported by Geant4. The \c is_geant_compatible method can be used to
 * warn the user.
 */
class ScintillationParams final : public ParamsDataInterface<ScintillationData>
{
  public:
    // Construct with optical materials and scintillation process data
    ScintillationParams(optical::MaterialParams const& optical_mat,
                        inp::ScintillationProcess const& process);

    // Whether any celeritas-only features are present
    bool is_geant_compatible() const;

    //! Access physics properties on the host
    HostRef const& host_ref() const final { return mirror_.host_ref(); }

    //! Access physics properties on the device
    DeviceRef const& device_ref() const final { return mirror_.device_ref(); }

  private:
    // Host/device storage and reference
    ParamsDataStore<ScintillationData> mirror_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
