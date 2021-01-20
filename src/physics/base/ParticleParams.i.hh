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
 * Get particle name.
 */
const std::string& ParticleParams::id_to_label(ParticleDefId id) const
{
    CELER_EXPECT(id < this->size());
    return md_[id.get()].first;
}

//---------------------------------------------------------------------------//
/*!
 * Get PDG code for a particle ID.
 */
PDGNumber ParticleParams::id_to_pdg(ParticleDefId id) const
{
    CELER_EXPECT(id < this->size());
    return md_[id.get()].second;
}

//---------------------------------------------------------------------------//
/*!
 * Find the ID from a name.
 */
ParticleDefId ParticleParams::find(const std::string& name) const
{
    auto iter = name_to_id_.find(name);
    if (iter == name_to_id_.end())
    {
        return ParticleDefId{};
    }
    return iter->second;
}

//---------------------------------------------------------------------------//
/*!
 * Find the ID from a PDG code.
 */
ParticleDefId ParticleParams::find(PDGNumber pdg_code) const
{
    auto iter = pdg_to_id_.find(pdg_code);
    if (iter == pdg_to_id_.end())
    {
        return ParticleDefId{};
    }
    return iter->second;
}

//---------------------------------------------------------------------------//
/*!
 * Get particle definitions based on an ID number.
 */
const ParticleDef& ParticleParams::get(ParticleDefId defid) const
{
    CELER_EXPECT(defid < host_defs_.size());
    return host_defs_[defid.get()];
}

//---------------------------------------------------------------------------//
} // namespace celeritas
