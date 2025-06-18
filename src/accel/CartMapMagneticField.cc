//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/CartMapMagneticField.cc
//---------------------------------------------------------------------------//

#include "CartMapMagneticField.hh"

#include <algorithm>
#include <CLHEP/Units/SystemOfUnits.h>
#include <G4MagneticField.hh>
#include <corecel/Assert.hh>

#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/data/HyperslabIndexer.hh"
#include "corecel/math/Quantity.hh"
#include "geocel/GeantGeoUtils.hh"
#include "geocel/g4/Convert.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/ext/GeantUnits.hh"
#include "celeritas/field/CartMapFieldInput.hh"
#include "celeritas/field/CartMapFieldParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Generates input for CartMapField params with configurable uniform grid
 * dimensions in native Geant4 units. This must be called after
 * G4RunManager::Initialize as it will retrieve the G4FieldManager's field
 * to sample it.
 */
CartMapFieldParams::Input
MakeCartMapFieldInput(CartMapFieldGridParams const& params)
{
    CELER_VALIDATE(params.max_x > params.min_x,
                   << "maximum X must be greater than minimum X");
    CELER_VALIDATE(params.max_y > params.min_y,
                   << "maximum Y must be greater than minimum Y");
    CELER_VALIDATE(params.max_z > params.min_z,
                   << "maximum Z must be greater than minimum Z");
    CELER_VALIDATE(params.num_x >= 2,
                   << "number of X grid points must be at least 2");
    CELER_VALIDATE(params.num_y >= 2,
                   << "number of Y grid points must be at least 2");
    CELER_VALIDATE(params.num_z >= 2,
                   << "number of Z grid points must be at least 2");

    CartMapFieldParams::Input field_input;

    // Convert from Geant4 units to native units
    field_input.min_x = convert_from_geant(params.min_x, clhep_length);
    field_input.max_x = convert_from_geant(params.max_x, clhep_length);
    field_input.num_x = params.num_x;

    field_input.min_y = convert_from_geant(params.min_y, clhep_length);
    field_input.max_y = convert_from_geant(params.max_y, clhep_length);
    field_input.num_y = params.num_y;

    field_input.min_z = convert_from_geant(params.min_z, clhep_length);
    field_input.max_z = convert_from_geant(params.max_z, clhep_length);
    field_input.num_z = params.num_z;

    size_type const total_points = params.num_x * params.num_y * params.num_z;
    field_input.field.resize(static_cast<size_type>(Axis::size_)
                             * total_points);

    Array<size_type, 4> const dims{params.num_x,
                                   params.num_y,
                                   params.num_z,
                                   static_cast<size_type>(Axis::size_)};
    HyperslabIndexer const flat_index{dims};

    G4Field const* g4field = celeritas::geant_field();
    CELER_VALIDATE(g4field,
                   << "no Geant4 global field has been set: cannot build "
                      "CartMapMagneticField");

    // Calculate grid spacing
    G4double const dx = (params.max_x - params.min_x) / (params.num_x - 1);
    G4double const dy = (params.max_y - params.min_y) / (params.num_y - 1);
    G4double const dz = (params.max_z - params.min_z) / (params.num_z - 1);

    Array<G4double, 3> bfield;
    for (size_type ix = 0; ix < params.num_x; ++ix)
    {
        G4double x = params.min_x + ix * dx;
        for (size_type iy = 0; iy < params.num_y; ++iy)
        {
            G4double y = params.min_y + iy * dy;
            for (size_type iz = 0; iz < params.num_z; ++iz)
            {
                G4double z = params.min_z + iz * dz;

                auto* cur_bfield = field_input.field.data()
                                   + flat_index(ix, iy, iz, 0);

                Array<G4double, 4> pos = {x, y, z, 0};
                g4field->GetFieldValue(pos.data(), bfield.data());

                // Convert field values from Geant4 units to native units
                auto bfield_native
                    = convert_from_geant(bfield.data(), clhep_field);
                std::copy(
                    bfield_native.cbegin(), bfield_native.cend(), cur_bfield);
            }
        }
    }
    CELER_ENSURE(field_input);
    return field_input;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
