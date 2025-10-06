//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/setup/Events.cc
//---------------------------------------------------------------------------//
#include "Events.hh"

#include <variant>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/cont/VariantUtils.hh"
#include "corecel/io/StringUtils.hh"
#include "corecel/sys/ScopedMem.hh"
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
auto read_events(EventReaderInterface&& generate)
{
    std::vector<std::vector<Primary>> result;
    auto event = generate();
    while (!event.empty())
    {
        result.push_back(event);
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
    ScopedMem record_mem("setup::events");
    ScopedProfiling profile_this{"setup::events"};

    return std::visit(
        Overload{
            [&particles](inp::CorePrimaryGenerator const& pg) {
                return read_events(PrimaryGenerator{pg, *particles});
            },
            [&particles](inp::SampleFileEvents const& sfe) {
                return read_events(RootEventSampler{sfe.event_file,
                                                    particles,
                                                    sfe.num_events,
                                                    sfe.num_merged,
                                                    sfe.seed});
            },
            [&particles](inp::ReadFileEvents const& rfe) {
                if (ends_with(rfe.event_file, ".jsonl"))
                {
                    return read_events(
                        JsonEventReader{rfe.event_file, particles});
                }
                else if (ends_with(rfe.event_file, ".root"))
                {
                    return read_events(
                        RootEventReader{rfe.event_file, particles});
                }
                else
                {
                    // Assume filename is one of the HepMC3-supported
                    // extensions
                    return read_events(EventReader{rfe.event_file, particles});
                }
            },
        },
        e);
}

//---------------------------------------------------------------------------//
}  // namespace setup
}  // namespace celeritas
