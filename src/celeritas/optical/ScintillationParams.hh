//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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

namespace optical
{

//---------------------------------------------------------------------------//
/*!
 * Build and manage scintillation data.
 *
 * When not imported from Geant4 (which uses
 *  \c G4OpticalParameters::GetScintByParticleType to select what data must be
 * stored), the manually constructed \c Input data must store *either* material
 * or particle data, never both.
 */
class ScintillationParams final : public ParamsDataInterface<ScintillationData>
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstParticles = std::shared_ptr<ParticleParams const>;
    using ScintillationDataCRef = HostCRef<ScintillationData>;
    //!@}

    //! Scintillation data for all materials and particles
    struct Input
    {
        using VecOptMatId = std::vector<OptMatId>;
        using VecSPId = std::vector<ScintillationParticleId>;

        std::vector<double> resolution_scale;

        //! Material-only spectra
        std::vector<ImportMaterialScintSpectrum> materials;

        //!< ParticleId to ScintillationParticleId
        VecSPId pid_to_scintpid;
        //! Particle and material spectra [ParticleScintSpectrumId]
        std::vector<ImportParticleScintSpectrum> particles;

        explicit operator bool() const
        {
            return (pid_to_scintpid.empty() == particles.empty())
                   && !resolution_scale.empty()
                   && (materials.empty() != particles.empty());
        }
    };

  public:
    // Construct with imported data and particle params
    static std::shared_ptr<ScintillationParams>
    from_import(ImportData const& data, SPConstParticles particle_params);

    // Construct with scintillation components
    explicit ScintillationParams(Input const& input);

    //! Access physics properties on the host
    HostRef const& host_ref() const final { return mirror_.host_ref(); }

    //! Access physics properties on the device
    DeviceRef const& device_ref() const final { return mirror_.device_ref(); }

  private:
    // Host/device storage and reference
    CollectionMirror<ScintillationData> mirror_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
