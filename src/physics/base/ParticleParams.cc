//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ParticleParams.cc
//---------------------------------------------------------------------------//
#include "ParticleParams.hh"

#include "base/Macros.hh"
#include "comm/Device.hh"
#include "celeritas_config.h"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with a vector of particle definitions.
 */
ParticleParams::ParticleParams(const Input& defs)
{
    md_.reserve(defs.size());
    host_defs_.reserve(defs.size());
    for (const auto& particle : defs)
    {
        CELER_EXPECT(!particle.name.empty());
        CELER_EXPECT(particle.pdg_code);
        CELER_EXPECT(particle.mass >= zero_quantity());
        CELER_EXPECT(particle.decay_constant >= 0);

        // Add host metadata
        ParticleId    id(name_to_id_.size());
        bool          inserted;
        std::tie(std::ignore, inserted)
            = name_to_id_.insert({particle.name, id});
        CELER_ASSERT(inserted);
        std::tie(std::ignore, inserted)
            = pdg_to_id_.insert({particle.pdg_code, id});
        CELER_ASSERT(inserted);

        // Save the metadata on the host
        md_.push_back({particle.name, particle.pdg_code});

        // Save the definitions on the host
        ParticleDef host_def;
        host_def.mass           = particle.mass;
        host_def.charge         = particle.charge;
        host_def.decay_constant = particle.decay_constant;
        host_defs_.push_back(std::move(host_def));
    }

    if (celeritas::is_device_enabled())
    {
        device_defs_ = DeviceVector<ParticleDef>{host_defs_.size()};
        device_defs_.copy_to_device(make_span(host_defs_));
        CELER_ENSURE(device_defs_.size() == defs.size());
    }

    CELER_ENSURE(md_.size() == defs.size());
    CELER_ENSURE(name_to_id_.size() == defs.size());
    CELER_ENSURE(pdg_to_id_.size() == defs.size());
    CELER_ENSURE(host_defs_.size() == defs.size());
}

//---------------------------------------------------------------------------//
/*!
 * Get a view to the managed host data for debugging.
 *
 * This should be primarily used by unit tests.
 */
ParticleParamsPointers ParticleParams::host_pointers() const
{
    ParticleParamsPointers result;
    result.defs = make_span(host_defs_);
    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get a view to the managed data.
 */
ParticleParamsPointers ParticleParams::device_pointers() const
{
    CELER_EXPECT(!device_defs_.empty());
    ParticleParamsPointers result;
    result.defs = device_defs_.device_pointers();
    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
