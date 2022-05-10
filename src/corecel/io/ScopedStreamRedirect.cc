//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/ScopedStreamRedirect.cc
//---------------------------------------------------------------------------//
#include "ScopedStreamRedirect.hh"

#include <cctype>

#include "corecel/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with pointer to a stream such as cout.
 */
ScopedStreamRedirect::ScopedStreamRedirect(std::ostream* os)
    : input_stream_(os)
{
    CELER_EXPECT(input_stream_);
    input_buffer_ = input_stream_->rdbuf();
    input_stream_->rdbuf(temp_stream_.rdbuf());
}

//---------------------------------------------------------------------------//
/*!
 * Restore stream on destruction.
 */
ScopedStreamRedirect::~ScopedStreamRedirect()
{
    input_stream_->rdbuf(input_buffer_);
}

//---------------------------------------------------------------------------//
/*!
 * Get redirected output with trailing whitespaces removed.
 */
std::string ScopedStreamRedirect::str()
{
    input_stream_->flush();

    std::string result = temp_stream_.str();
    while (!result.empty() && std::isspace(result.back()))
    {
        result.pop_back();
    }
    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
