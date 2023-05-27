//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/Stream.cc
//---------------------------------------------------------------------------//
#include "Stream.hh"

#include <algorithm>
#include <iostream>

#include "corecel/Assert.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct by creating a stream.
 */
Stream::Stream()
{
    CELER_DEVICE_CALL_PREFIX(StreamCreate(&stream_));
}

//---------------------------------------------------------------------------//
/*!
 * Destroy the stream.
 */
Stream::~Stream()
{
    if (stream_ != nullptr)
    {
        try
        {
            CELER_DEVICE_CALL_PREFIX(StreamDestroy(stream_));
        }
        catch (RuntimeError const& e)
        {
            std::cerr << "Failed to destroy stream: " << e.what() << std::endl;
        }
        catch (...)
        {
            std::cerr << "Failed to destroy stream" << std::endl;
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Move construct.
 */
Stream::Stream(Stream&& other) noexcept
{
    this->swap(other);
}

//---------------------------------------------------------------------------//
/*!
 * Move assign.
 */
Stream& Stream::operator=(Stream&& other) noexcept
{
    Stream temp(std::move(other));
    this->swap(temp);
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Swap.
 */
void Stream::swap(Stream& other) noexcept
{
    std::swap(stream_, other.stream_);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
