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
auto read_events(EventReaderInterface& generate, bool merge)
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

    using UP_ERI = std::unique_ptr<EventReaderInterface>;
    auto generator = std::visit(
        return_as<UP_ERI>(Overload{
            [&particles](inp::CorePrimaryGenerator const& pg) {
                return std::make_unique<PrimaryGenerator>(pg, *particles);
            },
            [&particles](inp::SampleFileEvents const& sfe) {
                return std::make_unique<RootEventSampler>(sfe.event_file,
                                                          particles,
                                                          sfe.num_events,
                                                          sfe.num_merged,
                                                          sfe.seed);
            },
            [&particles](inp::ReadFileEvents const& rfe) -> UP_ERI {
                if (ends_with(rfe.event_file, ".jsonl"))
                {
                    return std::make_unique<JsonEventReader>(rfe.event_file,
                                                             particles);
                }
                else if (ends_with(rfe.event_file, ".root"))
                {
                    return std::make_unique<RootEventReader>(rfe.event_file,
                                                             particles);
                }
                else
                {
                    // Assume filename has a HepMC3-supported extension
                    return std::make_unique<EventReader>(rfe.event_file,
                                                         particles);
                }
            },
        }),
        e.generator);

    return read_events(*generator, e.merge);
}

//---------------------------------------------------------------------------//
}  // namespace setup
}  // namespace celeritas
