//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/JsonEventWriter.cc
//---------------------------------------------------------------------------//
#include "JsonEventWriter.hh"

#include <nlohmann/json.hpp>

#include "corecel/Assert.hh"
#include "corecel/cont/ArrayIO.json.hh"
#include "corecel/io/Join.hh"
#include "corecel/io/JsonUtils.json.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/StringUtils.hh"
#include "corecel/math/QuantityIO.json.hh"
#include "celeritas/phys/ParticleParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with output filename.
 */
JsonEventWriter::JsonEventWriter(std::string const& filename,
                                 SPConstParticles particles)
    : outfile_(filename), particles_(std::move(particles))
{
    CELER_EXPECT(!filename.empty());
    CELER_EXPECT(outfile_);
    CELER_EXPECT(particles_);

    if (!ends_with(filename, ".jsonl"))
    {
        CELER_LOG(warning) << "JSON event writer expects a jsonl file";
    }
    CELER_LOG(info) << "Creating JSON event file at " << filename;
}

//---------------------------------------------------------------------------//
/*!
 * Write all primaries from a single event.
 */
void JsonEventWriter::operator()(VecPrimary const& primaries)
{
    CELER_EXPECT(!primaries.empty());

    // Increment contiguous event id
    EventId const event_id{event_count_++};

    nlohmann::json event;
    event["event_id"] = event_id;
    event["primaries"] = nlohmann::json::array();
    for (auto const& p : primaries)
    {
        if (!warned_mismatched_events_ && p.event_id != event_id)
        {
            CELER_LOG_LOCAL(warning)
                << R"(Event IDs will not match output: this is a known issue)";
            warned_mismatched_events_ = true;
        }

        auto j = nlohmann::json{
            {"pdg", particles_->id_to_pdg(p.particle_id).get()},
            {"energy", p.energy},
            {"position", p.position},
            {"direction", p.direction},
            {"time", p.time},
        };
        event["primaries"].push_back(j);
    }
    outfile_ << event.dump() << "\n";
    outfile_.flush();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
