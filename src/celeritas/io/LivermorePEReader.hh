//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/LivermorePEReader.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>

#include "celeritas/phys/AtomicNumber.hh"

#include "ImportLivermorePE.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Load the Livermore EPICS2014 photoelectric data.
 */
class LivermorePEReader
{
  public:
    //!@{
    //! \name Type aliases
    using result_type = ImportLivermorePE;
    //!@}

  public:
    // Construct the reader and locate the data using the environment variable
    LivermorePEReader();

    // Construct the reader from the path to the data directory
    explicit LivermorePEReader(char const* path);

    // Read the data for the given element
    result_type operator()(AtomicNumber atomic_number) const;

  private:
    // Directory containing the Livermore photoelectric data
    std::string path_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
