//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/ParticleParams.cc
//---------------------------------------------------------------------------//
#include "ParticleParams.hh"

#include <algorithm>
#include <cstdlib>
#include <tuple>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/ScopedMem.hh"
#include "celeritas/io/ImportData.hh"

#include "PDGNumber.hh"
#include "ParticleData.hh"  // IWYU pragma: associated
#include "ParticleView.hh"

#include "detail/ParticleInserter.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with imported data.
 */
std::shared_ptr<ParticleParams>
ParticleParams::from_import(ImportData const& data)
{
    CELER_EXPECT(!data.particles.empty());

    Input defs = data.particles;

    // Sort by increasing mass, then by PDG code (positive before negative of
    // the same absolute value). Placing lighter particles
    // (more likely to be created by various processes, so more "light
    // particle" tracks) together at the beginning of the list will make it
    // easier to human-read the particles while debugging, and having them
    // at adjacent memory locations could improve caching.
    auto to_particle_key = [](auto const& inp) {
        int pdg = inp.pdg.get();
        return std::make_tuple(inp.mass, std::abs(pdg), pdg < 0);
    };
    std::sort(defs.begin(),
              defs.end(),
              [to_particle_key](auto const& lhs, auto const& rhs) {
                  return to_particle_key(lhs) < to_particle_key(rhs);
              });

    return std::make_shared<ParticleParams>(std::move(defs));
}

//---------------------------------------------------------------------------//
/*!
 * Construct with a vector of particle definitions.
 */
ParticleParams::ParticleParams(Input const& input)
{
    ScopedMem record_mem("ParticleParams.construct");

    md_.reserve(input.size());

    // Build elements and materials on host.
    HostVal<ParticleParamsData> host_data;
    detail::ParticleInserter insert_particle(&host_data);
    for (auto const& particle : input)
    {
        CELER_VALIDATE(particle.pdg,
                       << "input particle '" << particle.name
                       << "' was not assigned a PDG code");
        CELER_EXPECT(!particle.name.empty());

        ParticleId id = insert_particle(particle);

        // Add host metadata
        md_.push_back({particle.name, particle.pdg});
        bool inserted = name_to_id_.insert({particle.name, id}).second;
        CELER_VALIDATE(inserted,
                       << "multiple particles share the name '"
                       << particle.name << "'");
        inserted = pdg_to_id_.insert({particle.pdg, id}).second;
        if (!inserted)
        {
            CELER_LOG(warning) << "multiple particles share the PDG code "
                               << particle.pdg.get();
        }
    }

    // Move to mirrored data, copying to device
    data_ = ParamsDataStore<ParticleParamsData>{std::move(host_data)};

    CELER_ENSURE(md_.size() == input.size());
    CELER_ENSURE(name_to_id_.size() == input.size());
    CELER_ENSURE(pdg_to_id_.size() == input.size());
    CELER_ENSURE(this->host_ref().size() == input.size());
}

//---------------------------------------------------------------------------//
/*!
 * Get particle properties in host code.
 */
ParticleView ParticleParams::get(ParticleId id) const
{
    CELER_EXPECT(id < this->host_ref().size());
    return ParticleView(this->host_ref(), id);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
