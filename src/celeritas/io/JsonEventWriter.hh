//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/JsonEventWriter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <fstream>
#include <string>

#include "corecel/Macros.hh"
#include "celeritas/phys/Primary.hh"

#include "EventIOInterface.hh"

namespace celeritas
{
class ParticleParams;

//---------------------------------------------------------------------------//
/*!
 * Export primary data to json.
 *
 * This write one event per line.
 */
class JsonEventWriter : public EventWriterInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstParticles = std::shared_ptr<ParticleParams const>;
    //!@}

    // Construct with output filename
    JsonEventWriter(std::string const& filename, SPConstParticles particles);

    //! Prevent copying and moving
    CELER_DELETE_COPY_MOVE(JsonEventWriter);
    ~JsonEventWriter() override = default;

    // Export primaries to json
    void operator()(VecPrimary const& primaries) override;

  private:
    std::ofstream outfile_;
    SPConstParticles particles_;
    EventId::size_type event_count_{0};
    bool warned_mismatched_events_{false};
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
