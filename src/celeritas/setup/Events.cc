//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/setup/Events.cc
//---------------------------------------------------------------------------//

namespace celeritas
{
namespace setup
{
//---------------------------------------------------------------------------//
/*!
 * Read events from a file or build using a primary generator.
 *
 * This returns the total number of events.
 */
size_type
Runner::build_events(RunnerInput const& inp, SPConstParticles particles)
{
    ScopedMem record_mem("Runner.build_events");

    if (inp.merge_events)
    {
        // All events will be transported simultaneously on a single stream
        events_.resize(1);
    }

    auto read_events = [&](auto&& generate) {
        auto event = generate();
        while (!event.empty())
        {
            if (inp.merge_events)
            {
                events_.front().insert(
                    events_.front().end(), event.begin(), event.end());
            }
            else
            {
                events_.push_back(event);
            }
            event = generate();
        }
        return generate.num_events();
    };

    if (inp.primary_options)
    {
        return read_events(
            PrimaryGenerator::from_options(particles, inp.primary_options));
    }
    else if (ends_with(inp.event_file, ".root"))
    {
        if (inp.file_sampling_options)
        {
            // Sampling options are assigned; use ROOT event sampler
            return read_events(
                RootEventSampler(inp.event_file,
                                 particles,
                                 inp.file_sampling_options.num_events,
                                 inp.file_sampling_options.num_merged,
                                 inp.seed));
        }
        else
        {
            // Use event reader
            return read_events(RootEventReader(inp.event_file, particles));
        }
    }
    else
    {
        // Assume filename is one of the HepMC3-supported extensions
        return read_events(EventReader(inp.event_file, particles));
    }
}

}  // namespace setup
}  // namespace celeritas
