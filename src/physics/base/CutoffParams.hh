//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file CutoffParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <map>
#include <vector>

#include "base/CollectionMirror.hh"
#include "physics/base/Units.hh"
#include "physics/base/ParticleParams.hh"
#include "physics/material/MaterialParams.hh"
#include "CutoffInterface.hh"

namespace celeritas
{
struct ImportData;

//---------------------------------------------------------------------------//
/*!
 * Data management for particle and material cutoff values.
 *
 * Geant4 provides accessors to its production cuts from its
 * \c G4MaterialCutsCouple class, which couples cutoff and material data.
 * During import, for simplicity, G4's production cuts are stored alongside
 * the material information, in \c ImportMaterial . Since this is a direct
 * import, the cutoff map in \c ImportMaterial stores only the cuts available
 * in Geant4, i.e. only values for gammas, electrons, positrons, and protons.
 *
 * In Celeritas, particle cutoff is stored contiguously in a single vector
 * of size num_particles * num_materials, which stores all particle cutoffs
 * for all materials. During import, any particle that is not in Geant4's
 * list receives a zero cutoff value. This opens the possibility to expand
 * cutoffs in the future, when data is not imported anymore.
 *
 * The \c Input structure provides a failsafe mechanism to construct the
 * host/device data.
 */
class CutoffParams
{
  public:
    //!@{
    //! References to constructed data
    using HostRef
        = CutoffParamsData<Ownership::const_reference, MemSpace::host>;
    using DeviceRef
        = CutoffParamsData<Ownership::const_reference, MemSpace::device>;

    using SPConstParticles = std::shared_ptr<const ParticleParams>;
    using SPConstMaterials = std::shared_ptr<const MaterialParams>;
    using MaterialCutoffs  = std::vector<ParticleCutoff>;
    //!@}

    //! Input data to construct this class
    struct Input
    {
        SPConstParticles                     particles;
        SPConstMaterials                     materials;
        std::map<PDGNumber, MaterialCutoffs> cutoffs;
    };

  public:
    // Construct from imported file
    static std::shared_ptr<CutoffParams> from_import(const ImportData& data);

    // Construct with cutoff input data
    explicit CutoffParams(const Input& input);

    //! Access cutoff data on the host
    const HostRef& host_pointers() const { return data_.host(); }

    //! Access cutoff data on the device
    const DeviceRef& device_pointers() const { return data_.device(); }

  private:
    // Host/device storage and reference
    CollectionMirror<CutoffParamsData> data_;
    using HostValue = CutoffParamsData<Ownership::value, MemSpace::host>;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
