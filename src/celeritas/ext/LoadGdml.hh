//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/LoadGdml.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>

#include "celeritas_config.h"
#include "corecel/Assert.hh"

class G4VPhysicalVolume;

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
//! Helper function to move destructor to .cc file for implementation hiding
struct PVDeleter
{
    void operator()(G4VPhysicalVolume*) const;
};

//---------------------------------------------------------------------------//
} // namespace detail

//---------------------------------------------------------------------------//
// TYPE ALIASES
//---------------------------------------------------------------------------//

//! Unique pointer with externally defined deleter
using UPG4PhysicalVolume
    = std::unique_ptr<G4VPhysicalVolume, detail::PVDeleter>;

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Load a GDML file and return the world volume.
 */
UPG4PhysicalVolume load_gdml(const std::string& gdml_filename);

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
#if !CELERITAS_USE_GEANT4
inline void detail::PVDeleter::operator()(G4VPhysicalVolume*) const
{
    CELER_ASSERT_UNREACHABLE();
}

inline UPG4PhysicalVolume load_gdml(const std::string&)
{
    CELER_NOT_CONFIGURED("Geant4");
}
#endif

//---------------------------------------------------------------------------//
} // namespace celeritas
