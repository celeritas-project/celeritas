//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/EventWriter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "celeritas_config.h"
#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/Types.hh"

namespace HepMC3
{
class Writer;
}

namespace celeritas
{
//---------------------------------------------------------------------------//
class ParticleParams;
struct Primary;

//---------------------------------------------------------------------------//
/*!
 * Write events using HepMC3.
 */
class EventWriter
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstParticles = std::shared_ptr<ParticleParams const>;
    using argument_type = std::vector<Primary> const&;
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
    CELER_DELETE_COPY_MOVE(EventWriter)

    // Write all the primaries from a single event
    void operator()(argument_type primaries);

  private:
    // Shared standard model particle data
    SPConstParticles particles_;

    // HepMC3 event record writer
    std::shared_ptr<HepMC3::Writer> writer_;

    // Number of events written
    EventId::size_type event_count_{0};
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
char const* to_cstring(EventWriter::Format);

//---------------------------------------------------------------------------//
}  // namespace celeritas
