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
 * These are named based on accessors in G4VEnergyLossProcess, with one
 * new table type, \c dedx_process, introduced to disambiguate the tables. In
 * Geant4, the \c dedx table belonging to the ionization process is actually
 * the sum of the de/dx for all processes that contribute to energy loss for
 * the given particle, while the \c dedx tables for the remaining processes are
 * the per-process energy loss. Here the tables are named to distinguish the
 * summed energy loss (\c dedx) from the energy loss for an individual process
 * (\c dedx_process). The \c ionization table is really just the \c
 * dedx_process table for ionization, so it is redundant. The \c range table is
 * calculated from the summed \c dedx table.
 */
enum class ImportTableType
{
    dedx,         //!< Energy loss summed over processes
    dedx_process, //!< Energy loss table for a process
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
