//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/JsonEventReader.cc
//---------------------------------------------------------------------------//
#include "JsonEventReader.hh"

#include <nlohmann/json.hpp>

#include "corecel/cont/ArrayIO.json.hh"
#include "corecel/io/Logger.hh"
#include "corecel/math/QuantityIO.json.hh"
#include "celeritas/phys/ParticleParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with input filename.
 */
JsonEventReader::JsonEventReader(std::string const& filename,
                                 SPConstParticles particles)
    : infile_(filename), particles_(std::move(particles))
{
    CELER_EXPECT(!filename.empty());

    CELER_VALIDATE(infile_,
                   << "failed to open file '" << filename << "' for reading");

    // Count the number of events in the file
    std::string line;
    while (std::getline(infile_, line))
    {
        ++num_events_;
    }

    // Reset stream to beginning
    infile_.clear();
    infile_.seekg(0, std::ios::beg);
}

//---------------------------------------------------------------------------//
/*!
 * Read single event from the file.
 */
auto JsonEventReader::operator()() -> result_type
{
    CELER_EXPECT(particles_);
    CELER_EXPECT(infile_);

    result_type result;

    std::string line;
    if (!std::getline(infile_, line))
    {
        // No more events
        return result;
    }
    auto event = nlohmann::json::parse(line);

    EventId event_id(event.at("event_id").get<EventId::size_type>());
    for (auto const& j : event.at("primaries"))
    {
        Primary p;
        p.event_id = event_id;
        p.particle_id = particles_->find((PDGNumber(j.at("pdg").get<int>())));
        j.at("energy").get_to(p.energy);
        j.at("position").get_to(p.position);
        j.at("direction").get_to(p.direction);
        p.time = j.at("time").get<real_type>();
        result.push_back(p);
    }

    CELER_ENSURE(!result.empty());
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
