//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/LivermorePEReader.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>

#include "celeritas/inp/Grid.hh"
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
    explicit LivermorePEReader(inp::Interpolation);

    // Construct the reader from the path to the data directory
    LivermorePEReader(char const* path, inp::Interpolation);

    // Read the data for the given element
    result_type operator()(AtomicNumber atomic_number) const;

  private:
    // Directory containing the Livermore photoelectric data
    std::string path_;
    // Interpolation method
    inp::Interpolation interpolation_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
