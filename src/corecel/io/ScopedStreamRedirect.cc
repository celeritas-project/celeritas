//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/ScopedStreamRedirect.cc
//---------------------------------------------------------------------------//
#include "ScopedStreamRedirect.hh"

#include "corecel/Assert.hh"
#include "corecel/sys/Environment.hh"

#include "Logger.hh"
#include "StringUtils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Whether signal handling is enabled.
 */
bool ScopedStreamRedirect::enabled()
{
    // Note that we negate the result to go from DISABLE to ENABLE
    static bool const result
        = !celeritas::getenv_flag("CELER_DISABLE_REDIRECT", false).value;
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct with pointer to a stream such as cout.
 */
ScopedStreamRedirect::ScopedStreamRedirect(std::ostream* os)
    : input_stream_(os)
{
    CELER_EXPECT(input_stream_);
    if (this->enabled())
    {
        input_buffer_ = input_stream_->rdbuf();
        input_stream_->rdbuf(temp_stream_.rdbuf());
    }
}

//---------------------------------------------------------------------------//
/*!
 * Restore stream on destruction.
 */
ScopedStreamRedirect::~ScopedStreamRedirect()
{
    if (input_buffer_)
    {
        input_stream_->rdbuf(input_buffer_);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Get redirected output with trailing whitespaces removed.
 *
 * If redirection is disabled, this will be an empty string.
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
 *
 * If redirection is disabled, this will be an empty stream.
 */
std::stringstream& ScopedStreamRedirect::get()
{
    input_stream_->flush();
    return temp_stream_;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
