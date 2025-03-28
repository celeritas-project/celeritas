//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/detail/UniqueFileNamer.cc
//---------------------------------------------------------------------------//
#include "UniqueFileNamer.hh"

#include <chrono>
#include <iomanip>
#include <sstream>

#include "corecel/Assert.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with original filename.
 */
UniqueFileNamer::UniqueFileNamer(std::string const& filename)
{
    CELER_EXPECT(!filename.empty());
    std::ostringstream stem;

    if (auto pos = filename.find_last_of('.');
        pos != std::string::npos && pos != 0)
    {
        // Has extension
        stem << filename.substr(0, pos);
        ext_ = filename.substr(pos);  // Include the dot
    }
    else
    {
        stem << filename;
    }

    // Append date+timestamp in format -YYMMDD-hhmmss
    auto tm_now = [] {
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        return *std::localtime(&time_t_now);
    }();
    stem << '-' << std::put_time(&tm_now, "%y%m%d-%H%M%S");
    stem_ = std::move(stem).str();

    CELER_ENSURE(!stem_.empty());
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
