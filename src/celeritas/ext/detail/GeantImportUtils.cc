//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantImportUtils.cc
//---------------------------------------------------------------------------//
#include "GeantImportUtils.hh"

#include <CLHEP/Units/SystemOfUnits.h>
#include <G4PhysicsTable.hh>
#include <G4PhysicsVectorType.hh>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"

namespace celeritas
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Get a multiplicative geant4-natural-units constant to convert the units.
 */
double units_to_scaling(ImportUnits units)
{
    using CLHEP::cm;
    using CLHEP::MeV;

    switch (units)
    {
        case ImportUnits::none:
            return 1;
        case ImportUnits::cm_inv:
            return cm;
        case ImportUnits::cm_mev_inv:
            return cm * MeV;
        case ImportUnits::mev:
            return 1 / MeV;
        case ImportUnits::mev_per_cm:
            return cm / MeV;
        case ImportUnits::cm:
            return 1 / cm;
        default:
            CELER_ASSERT_UNREACHABLE();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Convert physics vector type from Geant4 to Celeritas IO.
 *
 * Geant4 v11 has a different set of G4PhysicsVectorType enums.
 */
ImportPhysicsVectorType
to_import_physics_vector_type(G4PhysicsVectorType g4_vector_type)
{
    switch (g4_vector_type)
    {
#if CELERITAS_G4_V10
        case T_G4PhysicsVector:
            return ImportPhysicsVectorType::unknown;
#endif
        case T_G4PhysicsLinearVector:
            return ImportPhysicsVectorType::linear;
        case T_G4PhysicsLogVector:
#if CELERITAS_G4_V10
        case T_G4PhysicsLnVector:
#endif
            return ImportPhysicsVectorType::log;
        case T_G4PhysicsFreeVector:
#if CELERITAS_G4_V10
        case T_G4PhysicsOrderedFreeVector:
        case T_G4LPhysicsFreeVector:
#endif
            return ImportPhysicsVectorType::free;
    }
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
ImportPhysicsTable
import_table(G4PhysicsTable const& g4table, ImportTableType table_type)
{
    CELER_EXPECT(table_type != ImportTableType::size_);
    ImportPhysicsTable table;
    table.table_type = table_type;
    switch (table_type)
    {
        case ImportTableType::dedx:
        case ImportTableType::dedx_process:
        case ImportTableType::dedx_subsec:
        case ImportTableType::dedx_unrestricted:
        case ImportTableType::ionization_subsec:
            table.x_units = ImportUnits::mev;
            table.y_units = ImportUnits::mev_per_cm;
            break;
        case ImportTableType::csda_range:
        case ImportTableType::range:
        case ImportTableType::secondary_range:
            table.x_units = ImportUnits::mev;
            table.y_units = ImportUnits::cm;
            break;
        case ImportTableType::inverse_range:
            table.x_units = ImportUnits::cm;
            table.y_units = ImportUnits::mev;
            break;
        case ImportTableType::lambda:
        case ImportTableType::sublambda:
            table.x_units = ImportUnits::mev;
            table.y_units = ImportUnits::cm_inv;
            break;
        case ImportTableType::lambda_prim:
            table.x_units = ImportUnits::mev;
            table.y_units = ImportUnits::cm_mev_inv;
            break;
        default:
            CELER_ASSERT_UNREACHABLE();
    };

    // Convert units
    double x_scaling = units_to_scaling(table.x_units);
    double y_scaling = units_to_scaling(table.y_units);

    // Save physics vectors
    for (auto const* g4vector : g4table)
    {
        ImportPhysicsVector import_vec;

        // Populate ImportPhysicsVectors
        import_vec.vector_type
            = to_import_physics_vector_type(g4vector->GetType());
        import_vec.x.reserve(g4vector->GetVectorLength());
        import_vec.y.reserve(import_vec.x.size());

        for (auto j : celeritas::range(g4vector->GetVectorLength()))
        {
            import_vec.x.push_back(g4vector->Energy(j) * x_scaling);
            import_vec.y.push_back((*g4vector)[j] * y_scaling);
        }
        table.physics_vectors.push_back(std::move(import_vec));
    }

    return table;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
