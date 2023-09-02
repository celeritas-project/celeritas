//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/detail/StreamStorage.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"

#include "../Stream.hh"
#include "../ThreadId.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Helper class for managing CUDA/HIP streams.
 */
class StreamStorage
{
  public:
    //!@{
    //! \name Type aliases
    using VecStream = std::vector<Stream>;
    //!@}

  public:
    //! Construct with the default stream
    StreamStorage() = default;

    // Construct by creating the given number of streams
    explicit StreamStorage(size_type num_streams) : streams_(num_streams) {}

    //! Number of streams allocated
    size_type size() const { return streams_.size(); }

    // Access a stream
    inline Stream& get(StreamId);

  private:
    VecStream streams_;
    Stream default_stream_{nullptr};
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Access a stream.
 *
 * If no streams have been created the default stream is returned.
 */
Stream& StreamStorage::get(StreamId id)
{
    if (!streams_.empty())
    {
        CELER_ASSERT(id < streams_.size());
        return streams_[id.get()];
    }
    return default_stream_;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
