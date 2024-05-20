//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/RootEventSampler.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <random>

#include "celeritas/io/EventIOInterface.hh"
#include "celeritas/io/RootEventReader.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Use \c RootEventReader to load events and sample primaries from them.
 */
class RootEventSampler : public EventReaderInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstParticles = std::shared_ptr<ParticleParams const>;
    using UPRootEventReader = std::unique_ptr<RootEventReader>;
    using result_type = std::vector<Primary>;
    //!@}

  public:
    // Construct with input data for RootEventReader and sampling information
    RootEventSampler(std::string const& filename,
                     SPConstParticles particles,
                     size_type num_sampled_events,
                     size_type num_merged_events,
                     unsigned int seed);

    //! Sample primaries for a single event
    result_type operator()() final;

    //! Get total number of events
    size_type num_events() const final { return num_sampled_events_; }

  private:
    size_type num_sampled_events_;  // Total number of events
    size_type num_merged_events_;  // Number of events to be concatenated
    UPRootEventReader reader_;
    std::mt19937 rng_;
    std::uniform_int_distribution<size_type> select_event_;
    EventId event_count_{0};
};

//---------------------------------------------------------------------------//
#if !CELERITAS_USE_ROOT
inline RootEventSampler::RootEventSampler(
    std::string const&, SPConstParticles, size_type, size_type, unsigned int)
{
    CELER_DISCARD(num_sampled_events_);
    CELER_DISCARD(num_merged_events_);
    CELER_DISCARD(reader_);
    CELER_DISCARD(rng_);
    CELER_DISCARD(select_event_);
    CELER_DISCARD(event_count_);
    CELER_NOT_CONFIGURED("ROOT");
}

inline RootEventSampler::result_type RootEventSampler::operator()()
{
    CELER_ASSERT_UNREACHABLE();
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
