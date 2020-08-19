//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ParticleParams.cc
//---------------------------------------------------------------------------//
#include "ParticleParams.hh"

#include "base/Macros.hh"
#include "celeritas_config.h"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * \brief Construct with a vector of particle definitions
 */
ParticleParams::ParticleParams(const VecAnnotatedDefs& defs)
{
    host_defs_.reserve(defs.size());
    for (const auto& md_def : defs)
    {
        REQUIRE(!md_def.first.name.empty());
        REQUIRE(md_def.first.pdg_code);
        REQUIRE(md_def.second.mass >= 0);
        REQUIRE(md_def.second.decay_constant >= 0);

        // Add host metadata
        ParticleDefId id(name_to_id_.size());
        bool          inserted;
        std::tie(std::ignore, inserted)
            = name_to_id_.insert({md_def.first.name, id});
        CHECK(inserted);
        std::tie(std::ignore, inserted)
            = pdg_to_id_.insert({md_def.first.pdg_code, id});
        CHECK(inserted);

        // Save a copy of the definitions on the host
        host_defs_.push_back(md_def.second);
    }

#if CELERITAS_USE_CUDA
    device_defs_ = DeviceVector<ParticleDef>{host_defs_.size()};
    device_defs_.copy_to_device(make_span(host_defs_));
    ENSURE(device_defs_.size() == defs.size());
#endif
    ENSURE(name_to_id_.size() == defs.size());
    ENSURE(pdg_to_id_.size() == defs.size());
    ENSURE(host_defs_.size() == defs.size());
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
    ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get a view to the managed data.
 */
ParticleParamsPointers ParticleParams::device_pointers() const
{
    REQUIRE(!device_defs_.empty());
    ParticleParamsPointers result;
    result.defs = device_defs_.device_pointers();
    ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
