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
#include "celeritas/io/ImportOpticalMaterial.hh"

#include "ScintillationData.hh"

namespace celeritas
{
struct ImportData;

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

    //! Particle and material-dependent scintillation data.
    struct Input
    {
        // Indexed by MaterialId
        using VecImportScintSpectra = std::vector<ImportScintSpectrum>;

        // Indexed by ParticleId
        std::vector<VecImportScintSpectra> data;
    };

  public:
    // Construct with imported data
    static std::shared_ptr<ScintillationParams>
    from_import(ImportData const& data);

    // Construct with scintillation components
    explicit ScintillationParams(Input const& input);

    //! Access physics properties on the host
    HostRef const& host_ref() const final { return mirror_.host_ref(); }

    //! Access physics properties on the device
    DeviceRef const& device_ref() const final { return mirror_.device_ref(); }

  private:
    // Host/device storage and reference
    CollectionMirror<ScintillationData> mirror_;

    //// HELPER FUNCTIONS ////

    // Normalize yield probabilities and check correctness of populated data
    void
    normalize_and_validate(std::vector<ScintillationComponent>& vec_comp,
                           std::vector<ImportScintComponent> const& input_comp,
                           real_type const norm);
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
