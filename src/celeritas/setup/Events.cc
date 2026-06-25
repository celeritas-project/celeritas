//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/setup/Events.cc
//---------------------------------------------------------------------------//
#include "Events.hh"

#include <utility>
#include <variant>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/cont/VariantUtils.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/StringUtils.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "celeritas/io/EventIOInterface.hh"
#include "celeritas/io/EventReader.hh"
#include "celeritas/io/JsonEventReader.hh"
#include "celeritas/io/RootEventReader.hh"
#include "celeritas/phys/Primary.hh"
#include "celeritas/phys/PrimaryGenerator.hh"
#include "celeritas/phys/RootEventSampler.hh"

namespace celeritas
{
namespace setup
{
namespace
{
//---------------------------------------------------------------------------//
// NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
auto read_events(EventReaderInterface&& generate, bool merge)
{
    std::vector<std::vector<Primary>> result;
    auto event = generate();
    while (!event.empty())
    {
        if (merge)
        {
            if (result.empty())
            {
                result.emplace_back();
            }
            result.front().insert(result.front().end(),
                                  std::make_move_iterator(event.begin()),
                                  std::make_move_iterator(event.end()));
        }
        else
        {
            result.push_back(std::move(event));
        }
        event = generate();
    }
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Load events from a file.
 */
std::vector<std::vector<Primary>>
events(inp::Events const& e,
       std::shared_ptr<ParticleParams const> const& particles)
{
    CELER_EXPECT(particles);

    CELER_LOG(status) << "Loading events";
    ScopedProfiling profile_this{"setup::events"};

    return std::visit(
        Overload{
            [&](inp::CorePrimaryGenerator const& pg) {
                return read_events(PrimaryGenerator{pg, *particles}, e.merge);
            },
            [&](inp::SampleFileEvents const& sfe) {
                return read_events(RootEventSampler{sfe.event_file,
                                                    particles,
                                                    sfe.num_events,
                                                    sfe.num_merged,
                                                    sfe.seed},
                                   e.merge);
            },
            [&](inp::ReadFileEvents const& rfe) {
                if (ends_with(rfe.event_file, ".jsonl"))
                {
                    return read_events(
                        JsonEventReader{rfe.event_file, particles}, e.merge);
                }
                else if (ends_with(rfe.event_file, ".root"))
                {
                    return read_events(
                        RootEventReader{rfe.event_file, particles}, e.merge);
                }
                else
                {
                    // Assume filename is one of the HepMC3-supported
                    // extensions
                    return read_events(EventReader{rfe.event_file, particles},
                                       e.merge);
                }
            },
        },
        e.generator);
}

//---------------------------------------------------------------------------//
}  // namespace setup
}  // namespace celeritas
