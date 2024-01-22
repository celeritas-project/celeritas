//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/detail/OffloadWriter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <mutex>

#include "celeritas/io/EventIOInterface.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Dump primaries to a shared file, one event per flush.
 *
 * This thin class simply adds a mutex to the output writer for thread safety.
 */
class OffloadWriter
{
  public:
    //!@{
    //! \name Type aliases
    using UPWriter = std::unique_ptr<EventWriterInterface>;
    using argument_type = EventWriterInterface::argument_type;
    //!@}

  public:
    // Construct from a writer interface
    inline explicit OffloadWriter(UPWriter&& writer);

    // Write primaries
    inline void operator()(argument_type);

  private:
    std::mutex write_mutex_;
    UPWriter write_event_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from a writer.
 */
OffloadWriter::OffloadWriter(UPWriter&& writer)
    : write_event_{std::move(writer)}
{
}

//---------------------------------------------------------------------------//
/*!
 * Write primaries using a mutex.
 */
void OffloadWriter::operator()(argument_type primaries)
{
    std::lock_guard scoped_lock{write_mutex_};
    (*write_event_)(primaries);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
