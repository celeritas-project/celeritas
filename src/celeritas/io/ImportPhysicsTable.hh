//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportPhysicsTable.hh
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
    lambda,  //!< Macroscopic cross section
    lambda_prim,  //!< Cross section scaled by energy
    dedx,  //!< Energy loss summed over processes
    range,  //!< Integrated inverse energy loss
    msc_xs,  //!< Scaled transport cross section
    size_
};

//---------------------------------------------------------------------------//
/*!
 * Units of a physics table.
 */
enum class ImportUnits
{
    none,  //!< Unitless
    mev,  //!< Energy [MeV]
    mev_per_cm,  //!< Energy loss [MeV/cm]
    cm,  //!< Range [cm]
    cm_inv,  //!< Macroscopic xs [1/cm]
    cm_mev_inv,  //!< Macroscopic xs divided by energy [1/cm-MeV]
    mev_2_per_cm,  //!< Macroscopic xs with energy^2 factored in [MeV^2/cm]
    cm_2,  //!< Microscopic cross section [cm^2]
    size_
};

//---------------------------------------------------------------------------//
/*!
 * Imported physics table. Each table stores physics vectors for all materials.
 */
struct ImportPhysicsTable
{
    ImportTableType table_type{ImportTableType::size_};
    ImportUnits x_units{ImportUnits::none};
    ImportUnits y_units{ImportUnits::none};
    std::vector<ImportPhysicsVector> physics_vectors;

    explicit operator bool() const
    {
        return table_type != ImportTableType::size_ && !physics_vectors.empty();
    }
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

char const* to_cstring(ImportTableType value);
char const* to_cstring(ImportUnits value);

//---------------------------------------------------------------------------//
}  // namespace celeritas
