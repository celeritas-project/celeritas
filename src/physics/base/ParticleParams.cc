//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ParticleParams.cc
//---------------------------------------------------------------------------//
#include "ParticleParams.hh"

#include <algorithm>

#include "base/Assert.hh"
#include "base/CollectionBuilder.hh"
#include "io/ImportData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with imported data.
 */
std::shared_ptr<ParticleParams>
ParticleParams::from_import(const ImportData& data)
{
    CELER_EXPECT(data);

    Input defs(data.particles.size());

    for (auto i : range(data.particles.size()))
    {
        const auto& particle = data.particles.at(i);
        CELER_ASSERT(!particle.name.empty());

        // Convert metadata
        defs[i].name     = particle.name;
        defs[i].pdg_code = PDGNumber{particle.pdg};
        CELER_ASSERT(defs[i].pdg_code);

        // Convert data
        defs[i].mass           = units::MevMass{particle.mass};
        defs[i].charge         = units::ElementaryCharge{particle.charge};
        defs[i].decay_constant = (particle.is_stable
                                      ? ParticleDef::stable_decay_constant()
                                      : 1. / particle.lifetime);
    }

    // Sort by increasing mass, then by PDG code (positive before negative of
    // the same absolute value). Placing lighter particles
    // (more likely to be created by various processes, so more "light
    // particle" tracks) together at the beginning of the list will make it
    // easier to human-read the particles while debugging, and having them
    // at adjacent memory locations could improve cacheing.
    auto to_particle_key = [](const auto& inp) {
        int pdg = inp.pdg_code.get();
        return std::make_tuple(inp.mass, std::abs(pdg), pdg < 0);
    };
    std::sort(defs.begin(),
              defs.end(),
              [to_particle_key](const auto& lhs, const auto& rhs) {
                  return to_particle_key(lhs) < to_particle_key(rhs);
              });

    return std::make_shared<ParticleParams>(std::move(defs));
}

//---------------------------------------------------------------------------//
/*!
 * Construct with a vector of particle definitions.
 */
ParticleParams::ParticleParams(const Input& input)
{
    md_.reserve(input.size());

    // Build elements and materials on host.
    ParticleParamsData<Ownership::value, MemSpace::host> host_data;
    auto particles = make_builder(&host_data.particles);
    particles.reserve(input.size());

    for (const auto& particle : input)
    {
        CELER_EXPECT(!particle.name.empty());
        CELER_EXPECT(particle.mass >= zero_quantity());
        CELER_EXPECT(particle.decay_constant >= 0);

        // Add host metadata
        ParticleId id(name_to_id_.size());
        bool       inserted;
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
        particles.push_back(std::move(host_def));
    }

    // Move to mirrored data, copying to device
    data_ = CollectionMirror<ParticleParamsData>{std::move(host_data)};

    CELER_ENSURE(md_.size() == input.size());
    CELER_ENSURE(name_to_id_.size() == input.size());
    CELER_ENSURE(pdg_to_id_.size() == input.size());
    CELER_ENSURE(this->host_pointers().particles.size() == input.size());
}

//---------------------------------------------------------------------------//
} // namespace celeritas
