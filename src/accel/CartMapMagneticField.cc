//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/CartMapMagneticField.cc
//---------------------------------------------------------------------------//
#include "CartMapMagneticField.hh"

#include <algorithm>
#include <CLHEP/Units/SystemOfUnits.h>
#include <corecel/Assert.hh>

#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/math/ArrayUtils.hh"
#include "geocel/GeantGeoUtils.hh"
#include "geocel/g4/Convert.hh"
#include "celeritas/Types.hh"
#include "celeritas/ext/GeantUnits.hh"
#include "celeritas/field/CartMapField.hh"
#include "celeritas/field/CartMapFieldInput.hh"
#include "celeritas/field/CartMapFieldParams.hh"

#include "detail/MagneticFieldUtils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Generates input for CartMapField params with configurable uniform grid
 * dimensions in native Geant4 units using an explicit field.
 */
CartMapFieldParams::Input
MakeCartMapFieldInput(G4Field const& field,
                      CartMapFieldGridParams const& params)
{
    // Validate input parameters
    CELER_VALIDATE(params, << "invalid CartMapFieldGridParams provided");

    CartMapFieldParams::Input field_input;

    // Convert from Geant4 units to native units
    field_input.x.min = convert_from_geant(params.x.min, clhep_length);
    field_input.x.max = convert_from_geant(params.x.max, clhep_length);
    field_input.x.num = params.x.num;

    field_input.y.min = convert_from_geant(params.y.min, clhep_length);
    field_input.y.max = convert_from_geant(params.y.max, clhep_length);
    field_input.y.num = params.y.num;

    field_input.z.min = convert_from_geant(params.z.min, clhep_length);
    field_input.z.max = convert_from_geant(params.z.max, clhep_length);
    field_input.z.num = params.z.num;

    // Prepare field data storage
    size_type const total_points = params.x.num * params.y.num * params.z.num;
    field_input.field.resize(static_cast<size_type>(Axis::size_)
                             * total_points);

    Array<size_type, 4> const dims{params.x.num,
                                   params.y.num,
                                   params.z.num,
                                   static_cast<size_type>(Axis::size_)};

    // Calculate grid spacing
    G4double const dx = (params.x.max - params.x.min) / (params.x.num - 1);
    G4double const dy = (params.y.max - params.y.min) / (params.y.num - 1);
    G4double const dz = (params.z.max - params.z.min) / (params.z.num - 1);

    // Position calculator for Cartesian grid
    auto position_calculator = [&](size_type ix, size_type iy, size_type iz) {
        G4double x = params.x.min + ix * dx;
        G4double y = params.y.min + iy * dy;
        G4double z = params.z.min + iz * dz;
        return Array<G4double, 4>{x, y, z, 0};
    };

    // Field converter for Cartesian coordinates (no transformation needed)
    auto field_converter = [](Array<G4double, 3> const& bfield,
                              real_type* cur_bfield) {
        auto bfield_native = convert_from_geant(bfield.data(), clhep_field);
        std::copy(bfield_native.cbegin(), bfield_native.cend(), cur_bfield);
    };

    // Sample field using common utility
    detail::setup_and_sample_field(field,
                                   field_input.field.data(),
                                   dims,
                                   position_calculator,
                                   field_converter);

    CELER_ENSURE(field_input);
    return field_input;
}

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
    G4Field const* g4field = celeritas::geant_field();
    CELER_VALIDATE(g4field,
                   << "no Geant4 global field has been set: cannot build "
                      "magnetic field map");
    return MakeCartMapFieldInput(*g4field, params);
}

//---------------------------------------------------------------------------//
// CARTMAPMAGNETICFIELD IMPLEMENTATION
//---------------------------------------------------------------------------//

Real3 CartAdapterField::operator()(Real3 const& pos) const
{
    CartMapField calc_field{data};
    return calc_field(pos);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
