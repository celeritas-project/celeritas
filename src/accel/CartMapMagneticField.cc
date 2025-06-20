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
#include "corecel/math/ArrayUtils.hh"
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
// PIMPL IMPLEMENTATION
//---------------------------------------------------------------------------//

/*!
 * Implementation struct for CartMapMagneticField.
 *
 * This hides the C++20 dependency (CartMapField) from the header file.
 */
struct CartMapMagneticField::Impl
{
    CartMapMagneticField::SPConstFieldParams params;
    CartMapField calc_field;

    explicit Impl(CartMapMagneticField::SPConstFieldParams field_params)
        : params(std::move(field_params))
        , calc_field(CartMapField{params->ref<MemSpace::native>()})
    {
        CELER_EXPECT(params);
    }
};

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
    CELER_VALIDATE(params, << "invalid CartMapFieldGridParams provided");

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
    detail::setup_and_sample_field(
        field_input.field.data(), dims, position_calculator, field_converter);

    CELER_ENSURE(field_input);
    return field_input;
}

//---------------------------------------------------------------------------//
// CARTMAPMAGNETIC FIELD IMPLEMENTATION
//---------------------------------------------------------------------------//

/*!
 * Custom deleter implementation for PIMPL idiom.
 */
void CartMapMagneticField::ImplDeleter::operator()(Impl* ptr) const
{
    delete ptr;
}

//---------------------------------------------------------------------------//
/*!
 * Construct with the Celeritas shared CartMapFieldParams.
 */
CartMapMagneticField::CartMapMagneticField(SPConstFieldParams field_params)
    : pimpl_(new Impl(std::move(field_params)))
{
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the magnetic field vector at the given position.
 */
void CartMapMagneticField::GetFieldValue(G4double const pos[3],
                                         G4double* field) const
{
    // Calculate the magnetic field value in the native Celeritas unit system
    Real3 result = pimpl_->calc_field(convert_from_geant(pos, clhep_length));
    for (auto i = 0; i < 3; ++i)
    {
        // Return values of the field vector in CLHEP::tesla for Geant4
        auto ft = native_value_to<units::FieldTesla>(result[i]);
        field[i] = convert_to_geant(ft.value(), CLHEP::tesla);
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
