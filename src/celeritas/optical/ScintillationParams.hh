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
class ParticleParams;
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
    using SPConstParticles = std::shared_ptr<ParticleParams const>;
    using ScintillationDataCRef = HostCRef<ScintillationData>;
    //!@}

    //! Scintillation data for all materials
    struct Input
    {
        using VecOptMatId = std::vector<MaterialId>;
        using VecSPId = std::vector<ParticleId>;

        VecOptMatId optmatid_to_matid;  //!< OpticalMaterialId to MaterialId
        VecSPId scintpid_to_pid;  //!< ScintillationParticleId to ParticleId
        std::vector<ImportScintData> data;  //!< Indexed by OpticalMaterialId

        //! Whether all data are assigned and valid
        explicit operator bool() const
        {
            return !optmatid_to_matid.empty() && !data.empty();
        }
    };

  public:
    // Construct with imported data
    static std::shared_ptr<ScintillationParams>
    from_import(ImportData const& data, SPConstParticles particle_params);

    // Construct with scintillation components
    explicit ScintillationParams(Input const& input,
                                 SPConstParticles particle_params);

    //! Access physics properties on the host
    HostRef const& host_ref() const final { return mirror_.host_ref(); }

    //! Access physics properties on the device
    DeviceRef const& device_ref() const final { return mirror_.device_ref(); }

  private:
    // Host/device storage and reference
    CollectionMirror<ScintillationData> mirror_;

    //// HELPER FUNCTIONS ////

    // Convert imported scintillation components to Celeritas' components
    std::vector<ScintillationComponent>
    copy_components(std::vector<ImportScintComponent> const& input_comp);

    // Check correctness of populated component data
    void validate(std::vector<ScintillationComponent> const& vec_comp);
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
