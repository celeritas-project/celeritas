//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/NeutronXsReader.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>

#include "corecel/math/Quantity.hh"
#include "corecel/math/UnitUtils.hh"
#include "celeritas/UnitTypes.hh"
#include "celeritas/Units.hh"
#include "celeritas/phys/AtomicNumber.hh"

#include "ImportPhysicsVector.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Load the neutron elastic cross section (G4PARTICLEXSDATA/neutron/elZ) data.
 */
class NeutronXsReader
{
  public:
    //!@{
    //! \name Type aliases
    using result_type = ImportPhysicsVector;
    using MmSqMicroXs
        = Quantity<UnitProduct<units::Millimeter, units::Millimeter>, double>;
    //!@}

  public:
    // Construct the reader and locate the data using the environment variable
    NeutronXsReader();

    // Construct the reader from the path to the data directory
    explicit NeutronXsReader(char const* path);

    // Read the data for the given element
    result_type operator()(AtomicNumber atomic_number) const;

  private:
    // Directory containing the neutron elastic cross section data
    std::string path_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
