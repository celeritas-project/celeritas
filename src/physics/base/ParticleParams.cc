//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ParticleParams.cc
//---------------------------------------------------------------------------//
#include "ParticleParams.hh"

#include "base/Assert.hh"
#include "base/PieBuilder.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with a vector of particle definitions.
 */
ParticleParams::ParticleParams(const Input& input)
{
    md_.reserve(input.size());

    // Build elements and materials on host.
    ParticleParamsData<Ownership::value, MemSpace::host> host_data;
    auto particles = make_pie_builder(&host_data.particles);
    particles.reserve(input.size());

    for (const auto& particle : input)
    {
        CELER_EXPECT(!particle.name.empty());
        CELER_EXPECT(particle.pdg_code);
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
    data_ = PieMirror<ParticleParamsData>{std::move(host_data)};

    CELER_ENSURE(md_.size() == input.size());
    CELER_ENSURE(name_to_id_.size() == input.size());
    CELER_ENSURE(pdg_to_id_.size() == input.size());
    CELER_ENSURE(this->host_pointers().particles.size() == input.size());
}

//---------------------------------------------------------------------------//
} // namespace celeritas
