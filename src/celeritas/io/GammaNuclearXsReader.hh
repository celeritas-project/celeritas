//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/GammaNuclearXsReader.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>

#include "corecel/math/Quantity.hh"
#include "corecel/math/UnitUtils.hh"
#include "celeritas/UnitTypes.hh"
#include "celeritas/Units.hh"
#include "celeritas/inp/Grid.hh"
#include "celeritas/phys/AtomicNumber.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Load the gamma-nuclear cross section (G4PARTICLEXSDATA/gamma) data
 */
class GammaNuclearXsReader
{
  public:
    //!@{
    //! \name Type aliases
    using result_type = inp::Grid;
    using MmSqMicroXs
        = Quantity<UnitProduct<units::Millimeter, units::Millimeter>, double>;
    //!@}

  public:
    // Construct the reader using from the environment variable
    explicit GammaNuclearXsReader();

    // Construct the reader from the path to the data directory and the type
    GammaNuclearXsReader(char const* path);

    // Read the data for the given element
    result_type operator()(AtomicNumber atomic_number) const;

  private:
    // Directory containing the gamma-nuclear cross section data
    std::string path_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
