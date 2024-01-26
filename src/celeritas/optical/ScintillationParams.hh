//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ScintillationParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"

#include "ScintillationData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Build and manage scintillation data.
 */
class ScintillationParams final : public ParamsDataInterface<ScintillationData>
{
  public:
    //!@{
    //! \name Type aliases
    using ScintillationDataCRef = HostCRef<ScintillationData>;
    //!@}

    //! Material dependent scintillation components
    //! data[num_optical_materials][num_components]
    struct ScintillationInput
    {
        std::vector<std::vector<ScintillationComponent>> data;
    };

  public:
    // Construct with scintillation components
    explicit ScintillationParams(ScintillationInput const& input);

    //! Access physics properties on the host
    HostRef const& host_ref() const final { return mirror_.host_ref(); }

    //! Access physics properties on the device
    DeviceRef const& device_ref() const final { return mirror_.device_ref(); }

  private:
    // Host/device storage and reference
    CollectionMirror<ScintillationData> mirror_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
