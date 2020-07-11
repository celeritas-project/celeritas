//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ParticleParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "base/DeviceVector.hh"
#include "ParticleParamsPointers.hh"
#include "ParticleMd.hh"
#include "ParticleDef.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Data management for Standard Model particle classifications.
 *
 * This class represents "per-problem" shared data about standard
 * model particles being used.
 *
 * The ParticleParams is constructed on the host with a vector that
 * combines metadata (used for debugging output and interfacing with physics
 * setup) and data (used for on-device transport). Each entry in the
 * construction is assigned a unique \c ParticleDefId used for runtime access.
 */
class ParticleParams
{
  public:
    //@{
    //!
    using VecAnnotatedDefs = std::vector<std::pair<ParticleMd, ParticleDef>>;
    //@}

  public:
    // Construct with a vector of particle definitions
    explicit ParticleParams(const VecAnnotatedDefs& defs);

    // >>> HOST ACCESSORS

    //! Number of particle definitions
    size_type size() const { return name_to_id_.size(); }

    // Find the ID from a name
    inline ParticleDefId find(const std::string& name) const;

    // Find the ID from a PDG code
    inline ParticleDefId find(PDGNumber pdg_code) const;

    // Access definition on host for construction
    inline const ParticleDef& get(ParticleDefId id) const;

    // TESTING ONLY: Get a view to the managed data
    ParticleParamsPointers host_pointers() const;

    // >>> DEVICE ACCESSORS

    // Get a view to the managed data
    ParticleParamsPointers device_pointers() const;

  private:
    // Map particle names to registered IDs
    std::map<std::string, ParticleDefId> name_to_id_;

    // Map particle codes to registered IDs
    std::map<PDGNumber, ParticleDefId> pdg_to_id_;

    // Host copy of definitions for host construction of other classes
    std::vector<ParticleDef> host_defs_;

    // Particle definitions on device
    DeviceVector<ParticleDef> device_defs_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "ParticleParams.i.hh"

//---------------------------------------------------------------------------//
