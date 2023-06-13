//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/ScopedStreamRedirect.cc
//---------------------------------------------------------------------------//
#include "ScopedStreamRedirect.hh"

#include "corecel/Assert.hh"
#include "corecel/io/StringUtils.hh"

#include "StringUtils.hh"

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
    std::string result = this->get().str();
    while (!result.empty() && is_ignored_trailing(result.back()))
    {
        result.pop_back();
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get the raw stream after flushing the input.
 */
std::stringstream& ScopedStreamRedirect::get()
{
    input_stream_->flush();
    return temp_stream_;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
