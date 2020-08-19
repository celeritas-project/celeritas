//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ParticleParams.i.hh
//---------------------------------------------------------------------------//

#include "base/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Find the ID from a name.
 */
ParticleDefId ParticleParams::find(const std::string& name) const
{
    auto iter = name_to_id_.find(name);
    REQUIRE(iter != name_to_id_.end());
    return iter->second;
}

//---------------------------------------------------------------------------//
/*!
 * Find the ID from a PDG code.
 */
ParticleDefId ParticleParams::find(PDGNumber pdg_code) const
{
    auto iter = pdg_to_id_.find(pdg_code);
    REQUIRE(iter != pdg_to_id_.end());
    return iter->second;
}

//---------------------------------------------------------------------------//
/*!
 * Get particle definitions based on an ID number.
 */
const ParticleDef& ParticleParams::get(ParticleDefId defid) const
{
    REQUIRE(defid < host_defs_.size());
    return host_defs_[defid.get()];
}

//---------------------------------------------------------------------------//
} // namespace celeritas
