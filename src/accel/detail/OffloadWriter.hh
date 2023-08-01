//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/detail/OffloadWriter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <mutex>
#include <string>

#include "celeritas/io/EventWriter.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Dump primaries to a shared file, one event per flush.
 */
class OffloadWriter
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstParticles = std::shared_ptr<ParticleParams const>;
    using argument_type = std::vector<Primary> const&;
    //!@}

  public:
    // Construct with defaults
    inline explicit OffloadWriter(std::string const& filename,
                                  SPConstParticles const& particles);

    // Write primaries
    inline void operator()(argument_type);

  private:
    std::mutex write_mutex_;
    EventWriter write_event_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from a filename.
 */
OffloadWriter::OffloadWriter(std::string const& filename,
                             SPConstParticles const& particles)
    : write_event_{filename, particles}
{
}

//---------------------------------------------------------------------------//
/*!
 * Write primaries using a mutex.
 */
void OffloadWriter::operator()(argument_type primaries)
{
    std::lock_guard scoped_lock{write_mutex_};
    write_event_(primaries);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
