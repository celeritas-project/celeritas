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
const std::string& ParticleParams::id_to_label(ParticleId id) const
{
    CELER_EXPECT(id < this->size());
    return md_[id.get()].first;
}

//---------------------------------------------------------------------------//
/*!
 * Get PDG code for a particle ID.
 */
PDGNumber ParticleParams::id_to_pdg(ParticleId id) const
{
    CELER_EXPECT(id < this->size());
    return md_[id.get()].second;
}

//---------------------------------------------------------------------------//
/*!
 * Find the ID from a name.
 */
ParticleId ParticleParams::find(const std::string& name) const
{
    auto iter = name_to_id_.find(name);
    if (iter == name_to_id_.end())
    {
        return ParticleId{};
    }
    return iter->second;
}

//---------------------------------------------------------------------------//
/*!
 * Find the ID from a PDG code.
 */
ParticleId ParticleParams::find(PDGNumber pdg_code) const
{
    auto iter = pdg_to_id_.find(pdg_code);
    if (iter == pdg_to_id_.end())
    {
        return ParticleId{};
    }
    return iter->second;
}

//---------------------------------------------------------------------------//
/*!
 * Get material properties for the given material.
 */
ParticleView ParticleParams::get(ParticleId id) const
{
    CELER_EXPECT(id < this->host_pointers().particles.size());
    return ParticleView(this->host_pointers(), id);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
