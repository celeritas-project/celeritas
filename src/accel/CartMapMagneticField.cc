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

#include "detail/MagneticFieldUtils.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//

/*!
 * Validate grid dimension parameters.
 *
 * \param min_val Minimum coordinate value
 * \param max_val Maximum coordinate value
 * \param num_points Number of grid points
 * \param dim_name Name of dimension for error messages
 */
void validate_grid_dimension(G4double min_val,
                             G4double max_val,
                             size_type num_points,
                             char const* dim_name)
{
    CELER_VALIDATE(max_val > min_val,
                   << "maximum " << dim_name
                   << " must be greater than minimum " << dim_name);
    CELER_VALIDATE(num_points >= 2,
                   << "number of " << dim_name
                   << " grid points must be at least 2");
}

//---------------------------------------------------------------------------//
}  // namespace

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
    // Validate input parameters
    validate_grid_dimension(params.min_x, params.max_x, params.num_x, "X");
    validate_grid_dimension(params.min_y, params.max_y, params.num_y, "Y");
    validate_grid_dimension(params.min_z, params.max_z, params.num_z, "Z");

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

    // Prepare field data storage
    size_type const total_points = params.num_x * params.num_y * params.num_z;
    field_input.field.resize(static_cast<size_type>(Axis::size_)
                             * total_points);

    Array<size_type, 4> const dims{params.num_x,
                                   params.num_y,
                                   params.num_z,
                                   static_cast<size_type>(Axis::size_)};

    // Calculate grid spacing
    G4double const dx = (params.max_x - params.min_x) / (params.num_x - 1);
    G4double const dy = (params.max_y - params.min_y) / (params.num_y - 1);
    G4double const dz = (params.max_z - params.min_z) / (params.num_z - 1);

    // Position calculator for Cartesian grid
    auto position_calculator = [&](size_type ix, size_type iy, size_type iz) {
        G4double x = params.min_x + ix * dx;
        G4double y = params.min_y + iy * dy;
        G4double z = params.min_z + iz * dz;
        return Array<G4double, 4>{x, y, z, 0};
    };

    // Field converter for Cartesian coordinates (no transformation needed)
    auto field_converter = [](Array<G4double, 3> const& bfield,
                              real_type* cur_bfield) {
        auto bfield_native = convert_from_geant(bfield.data(), clhep_field);
        std::copy(bfield_native.cbegin(), bfield_native.cend(), cur_bfield);
    };

    // Sample field using common utility
    setup_and_sample_field(
        field_input.field.data(), dims, position_calculator, field_converter);

    CELER_ENSURE(field_input);
    return field_input;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
