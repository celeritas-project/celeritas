//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/EventWriter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>

#include "corecel/Config.hh"

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/Types.hh"

#include "EventIOInterface.hh"

namespace HepMC3
{
class Writer;
}

namespace celeritas
{
//---------------------------------------------------------------------------//
class ParticleParams;

//---------------------------------------------------------------------------//
/*!
 * Write events using HepMC3.
 */
class EventWriter : public EventWriterInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstParticles = std::shared_ptr<ParticleParams const>;
    //!@}

    //! Output format
    enum class Format
    {
        hepevt,
        hepmc2,
        hepmc3,
        size_
    };

  public:
    // Construct by parsing the extension
    EventWriter(std::string const& filename, SPConstParticles params);

    // Construct with a filename, particle data, and output format
    EventWriter(std::string const& filename,
                SPConstParticles params,
                Format fmt);

    //! Prevent copying and moving due to file ownership
    CELER_DELETE_COPY_MOVE(EventWriter);

    ~EventWriter() override = default;

    // Write all the primaries from a single event
    void operator()(VecPrimary const& primaries) final;

  private:
    // Shared standard model particle data
    SPConstParticles particles_;

    Format fmt_;

    // HepMC3 event record writer
    std::shared_ptr<HepMC3::Writer> writer_;

    // Number of events written
    EventId::size_type event_count_{0};

    bool warned_mismatched_events_{false};
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
char const* to_cstring(EventWriter::Format);

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
#if !CELERITAS_USE_HEPMC3
inline EventWriter::EventWriter(std::string const& s, SPConstParticles p)
    : EventWriter{s, p, Format::size_}
{
}
inline EventWriter::EventWriter(std::string const&, SPConstParticles, Format)
{
    CELER_DISCARD(particles_);
    CELER_DISCARD(fmt_);
    CELER_DISCARD(writer_);
    CELER_DISCARD(event_count_);
    CELER_DISCARD(warned_mismatched_events_);
    CELER_NOT_CONFIGURED("HepMC3");
}

inline void EventWriter::operator()(argument_type)
{
    CELER_ASSERT_UNREACHABLE();
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
