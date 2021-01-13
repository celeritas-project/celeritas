//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LivermoreParamsReader.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include "physics/em/LivermoreParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Load the Livermore EPICS2014 photoelectric data.
 */
class LivermoreParamsReader
{
  public:
    //!@{
    //! Type aliases
    using result_type = LivermoreParams::ElementInput;
    //!@}

  public:
    // Construct the reader and locate the data using the environment variable
    LivermoreParamsReader();

    // Construct the reader from the path to the data directory
    explicit LivermoreParamsReader(const char* path);

    // Read the data for the given element
    result_type operator()(int atomic_number) const;

  private:
    // Directory containing the Livermore photoelectric data
    std::string path_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
