//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/EventReader.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "celeritas_config.h"
#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"

#include "EventIOInterface.hh"

namespace HepMC3
{
class Reader;
}

namespace celeritas
{
//---------------------------------------------------------------------------//
class ParticleParams;
struct Primary;

//---------------------------------------------------------------------------//
/*!
 * Read a HepMC3 event record file and create primary particles.
 *
 * Each \c operator() call returns a vector of primaries from a single event
 * until all events have been read. Supported formats are Asciiv3, IO_GenEvent,
 * HEPEVT, and LHEF.
 *
 * \todo Define ImportPrimary with double precision.
 */
class EventReader : public EventReaderInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstParticles = std::shared_ptr<ParticleParams const>;
    using result_type = std::vector<Primary>;
    //!@}

  public:
    // Construct from a filename
    EventReader(std::string const& filename, SPConstParticles particles);

    //! Prevent copying and moving
    CELER_DELETE_COPY_MOVE(EventReader);

    // Read a single event from the event record
    result_type operator()() final;

    //! Get total number of events
    size_type num_events() const final { return num_events_; }

  private:
    using SPReader = std::shared_ptr<HepMC3::Reader>;

    // Shared standard model particle data
    SPConstParticles particles_;

    // HepMC3 event record reader
    SPReader reader_;

    // Number of events read
    size_type event_count_{0};

    // Total number of events in file
    size_type num_events_;
};

//---------------------------------------------------------------------------//
// Set verbosity from the environment (HEPMC3_VERBOSE)
void set_hepmc3_verbosity_from_env();

//---------------------------------------------------------------------------//
// Wrapper function for HepMC3::deduce_reader to avoid duplicate symbols
std::shared_ptr<HepMC3::Reader> open_hepmc3(std::string const& filename);

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
#if !CELERITAS_USE_HEPMC3
inline EventReader::EventReader(std::string const&, SPConstParticles)
{
    CELER_DISCARD(particles_);
    CELER_DISCARD(reader_);
    CELER_DISCARD(event_count_);
    CELER_DISCARD(num_events_);
    CELER_NOT_CONFIGURED("HepMC3");
}

inline auto EventReader::operator()() -> result_type
{
    CELER_ASSERT_UNREACHABLE();
}

inline void set_hepmc3_verbosity_from_env() {}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
