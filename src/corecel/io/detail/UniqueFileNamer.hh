//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/detail/UniqueFileNamer.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Helper class for generating unique filenames.
 *
 * Based on the timestamp when instantiated, it returns a filename formatted
 * like \c STEM-yymmdd-HHMMSS-COUNT.EXT .
 */
class UniqueFileNamer
{
  public:
    // Construct with original filename and split into stem and extension
    explicit UniqueFileNamer(std::string const& filename);

    //! Get the next filename in sequence
    std::string operator()()
    {
        return stem_ + "-" + std::to_string(++counter_) + ext_;
    }

  private:
    std::string stem_;  //!< Filename stem (without extension)
    std::string ext_;  //!< File extension (with dot)
    int counter_{0};  //!< Counter for unique names
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
