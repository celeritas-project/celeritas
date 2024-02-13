//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportPhysicsTable.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "celeritas/Units.hh"

#include "ImportPhysicsVector.hh"
#include "ImportUnits.hh"

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
    lambda,  //!< Macroscopic cross section
    lambda_prim,  //!< Cross section scaled by energy
    dedx,  //!< Energy loss summed over processes
    range,  //!< Integrated inverse energy loss
    msc_xs,  //!< Scaled transport cross section
    size_
};

//---------------------------------------------------------------------------//
/*!
 * Imported physics table. Each table stores physics vectors for all materials.
 */
struct ImportPhysicsTable
{
    ImportTableType table_type{ImportTableType::size_};
    ImportUnits x_units{ImportUnits::unitless};
    ImportUnits y_units{ImportUnits::unitless};
    std::vector<ImportPhysicsVector> physics_vectors;

    explicit operator bool() const
    {
        return table_type != ImportTableType::size_ && !physics_vectors.empty();
    }
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

// Get the string value for a table type
char const* to_cstring(ImportTableType value);

//---------------------------------------------------------------------------//
}  // namespace celeritas
