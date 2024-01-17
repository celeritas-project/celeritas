//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/SeltzerBergerReader.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>

#include "celeritas/phys/AtomicNumber.hh"

#include "ImportSBTable.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Read Seltzer-Berger data from Geant4's $G4LEDATA files.
 *
 * Use \c operator() to retrieve data for different atomic numbers.
 *
 * \code
    SeltzerBergerReader sb_reader();
    auto sb_data_vector = sb_reader(1); // Hydrogen
   \endcode
 */
class SeltzerBergerReader
{
  public:
    //!@{
    //! \name Type aliases
    using result_type = ImportSBTable;
    //!@}

  public:
    // Construct using $G4LEDATA
    SeltzerBergerReader();

    // Construct from a user defined path
    explicit SeltzerBergerReader(char const* path);

    // Read data from ascii for the given element
    result_type operator()(AtomicNumber atomic_number) const;

  private:
    std::string path_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
