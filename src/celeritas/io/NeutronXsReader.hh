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
 * Types of microscopic cross sections in G4PARTICLEXSDATA/neutron data.
 */
enum class NeutronXsType
{
    cap,  //!< Capture cross section
    el,  //!< Elastic cross section
    inel,  //!< Inelastic cross section
    size_
};

//---------------------------------------------------------------------------//
/*!
 * Load the neutron cross section (G4PARTICLEXSDATA/neutron) data by the
 * interaction type (capture, elastic, and inelastic).
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
    explicit NeutronXsReader(NeutronXsType type);

    // Construct the reader from the path to the data directory and the type
    explicit NeutronXsReader(char const* path, NeutronXsType type);

    // Read the data for the given element
    result_type operator()(AtomicNumber atomic_number) const;

  private:
    // Get the string value for the cross section data type

  private:
    // Directory containing the neutron elastic cross section data
    std::string path_;
    NeutronXsType type_;
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

// Get the string value for a neutron cross section type
char const* to_cstring(NeutronXsType value);

//---------------------------------------------------------------------------//
}  // namespace celeritas
