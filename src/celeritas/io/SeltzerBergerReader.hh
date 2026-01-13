//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/SeltzerBergerReader.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>

#include "corecel/inp/Grid.hh"
#include "celeritas/phys/AtomicNumber.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Read microscopic Brems cross sections from Geant4's $G4LEDATA files.
 *
 * Each call to this function-like class loads a single element.
 *
 * \code
    SeltzerBergerReader sb_reader();
    auto sb_data_vector = sb_reader(AtomicNumber{1}); // Hydrogen
   \endcode
 */
class SeltzerBergerReader
{
  public:
    //!@{
    //! \name Type aliases
    using result_type = inp::TwodGrid;
    //!@}

  public:
    // Construct using $G4LEDATA
    SeltzerBergerReader();

    // Construct from a user defined path
    explicit SeltzerBergerReader(std::string path);

    // Read data from ascii for the given element
    result_type operator()(AtomicNumber atomic_number) const;

  private:
    std::string path_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
