//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/ParticleParamsOutput.cc
//---------------------------------------------------------------------------//
#include "ParticleParamsOutput.hh"

#include <utility>

#include "celeritas_config.h"
#include "corecel/io/JsonPimpl.hh"
#include "celeritas/phys/Process.hh"

#include "ParticleParams.hh"  // IWYU pragma: keep
#if CELERITAS_USE_JSON
#    include <nlohmann/json.hpp>
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from shared particle data.
 */
ParticleParamsOutput::ParticleParamsOutput(SPConstParticleParams particles)
    : particles_(std::move(particles))
{
    CELER_EXPECT(particles_);
}

//---------------------------------------------------------------------------//
/*!
 * Write output to the given JSON object.
 */
void ParticleParamsOutput::output(JsonPimpl* j) const
{
#if CELERITAS_USE_JSON
    using json = nlohmann::json;
    auto label = json::array();
    auto pdg = json::array();

    for (auto id : range(ParticleId{particles_->size()}))
    {
        label.push_back(particles_->id_to_label(id));
        pdg.push_back(particles_->id_to_pdg(id).unchecked_get());
    }
    j->obj = {
        {"label", std::move(label)},
        {"pdg", std::move(pdg)},
    };
#else
    (void)sizeof(j);
#endif
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
