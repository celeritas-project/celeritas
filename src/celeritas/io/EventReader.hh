//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/EventReader.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "corecel/Types.hh"

namespace HepMC3
{
class Reader;
}

namespace celeritas
{
class ParticleParams;
struct Primary;

//---------------------------------------------------------------------------//
/*!
 * Read a HepMC3 event record file and create primary particles.
 *
 * Each \c operator() call returns a vector of primaries from a single event
 * until all events have been read. Supported formats are Asciiv3, IO_GenEvent,
 * HEPEVT, and LHEF.
 */
class EventReader
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstParticles = std::shared_ptr<const ParticleParams>;
    using result_type      = std::vector<Primary>;
    //!@}

  public:
    // Construct from a filename
    EventReader(const char* filename, SPConstParticles params);

    // Default destructor in .cc
    ~EventReader();

    // Read a single event from the event record
    result_type operator()();

  private:
    // Shared standard model particle data
    SPConstParticles params_;

    // HepMC3 event record reader
    std::shared_ptr<HepMC3::Reader> input_file_;

    // Number of events read
    size_type event_count_{0};
};

//---------------------------------------------------------------------------//
} // namespace celeritas
