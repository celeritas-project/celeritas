//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ImportPhysicsTable.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>
#include "ImportPhysicsVector.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Property being described by the physics table.
 *
 * These are named based on accessors in G4VEnergyLossProcess.
 */
enum class ImportTableType
{
    dedx,
    dedx_subsec,
    dedx_unrestricted,
    ionization,
    ionization_subsec,
    csda_range, //!< Continuous slowing down approximation
    range,
    secondary_range,
    inverse_range, //!< Inverse mapping of range: (range -> energy)
    lambda,        //!< Macroscopic cross section
    sublambda,     //!< For subcutoff regions
    lambda_prim,   //!< Cross section scaled by energy
};

//---------------------------------------------------------------------------//
/*!
 * Units of a physics table.
 */
enum class ImportUnits
{
    none,       //!< Unitless
    mev,        //!< Energy [MeV]
    mev_per_cm, //!< Energy loss [MeV/cm]
    cm,         //!< Range [cm]
    cm_inv,     //!< Macroscopic xs [1/cm]
    cm_mev_inv, //!< Macroscopic xs divided by energy [1/cm-MeV]
};

//---------------------------------------------------------------------------//
/*!
 * Imported physics table.
 *
 * The geant-exporter app stores Geant4 physics tables into a ROOT file, while
 * the RootImporter class is responsible for loading said data into memory.
 */
struct ImportPhysicsTable
{
    ImportTableType                  table_type;
    ImportUnits                      x_units;
    ImportUnits                      y_units;
    std::vector<ImportPhysicsVector> physics_vectors;
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

const char* to_cstring(ImportTableType value);
const char* to_cstring(ImportUnits value);

//---------------------------------------------------------------------------//
} // namespace celeritas
