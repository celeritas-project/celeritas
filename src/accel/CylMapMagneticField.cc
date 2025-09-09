//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/CylMapMagneticField.cc
//---------------------------------------------------------------------------//

#include "CylMapMagneticField.hh"

#include <algorithm>
#include <cmath>
#include <CLHEP/Units/SystemOfUnits.h>
#include <G4MagneticField.hh>
#include <corecel/Assert.hh>

#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/cont/EnumArray.hh"
#include "corecel/math/Quantity.hh"
#include "corecel/math/Turn.hh"
#include "geocel/g4/Convert.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/ext/GeantUnits.hh"
#include "celeritas/field/CylMapFieldInput.hh"
#include "celeritas/field/CylMapFieldParams.hh"

#include "detail/MagneticFieldUtils.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//

//! Cartesian to cylindrical 3D vector conversion.
inline void cartesian_to_cylindrical(Array<G4double, 3> const& cart,
                                     EnumArray<CylAxis, G4double>& cyl)
{
    double const phi = std::atan2(cart[1], cart[0]);
    cyl[CylAxis::r] = cart[0] * std::cos(phi) + cart[1] * std::sin(phi);
    cyl[CylAxis::phi] = -cart[0] * std::sin(phi) + cart[1] * std::cos(phi);
    cyl[CylAxis::z] = cart[2];
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Generates input for CylMapField params with configurable nonuniform grid
 * dimensions in native Geant4 units, and \f$\phi\f$ should be in the range
 * [0;\f$2\times\pi\f$]. This must be called after G4RunManager::Initialize as
 * it will retrieve the G4FieldManager's field to sample it.
 */
CylMapFieldParams::Input
MakeCylMapFieldInput(std::vector<G4double> const& r_grid,
                     std::vector<G4double> const& phi_values,
                     std::vector<G4double> const& z_grid)
{
    CylMapFieldParams::Input field_input;
    field_input.grid_r.reserve(r_grid.size());
    field_input.grid_phi.reserve(phi_values.size());
    field_input.grid_z.reserve(z_grid.size());

    // Convert from geant units to native units
    std::transform(r_grid.cbegin(),
                   r_grid.cend(),
                   std::back_inserter(field_input.grid_r),
                   [](auto r) { return convert_from_geant(r, clhep_length); });
    //  Convert phi values to Turn type
    std::transform(phi_values.cbegin(),
                   phi_values.cend(),
                   std::back_inserter(field_input.grid_phi),
                   [](auto phi) { return native_value_to<RealTurn>(phi); });
    std::transform(z_grid.cbegin(),
                   z_grid.cend(),
                   std::back_inserter(field_input.grid_z),
                   [](auto z) { return convert_from_geant(z, clhep_length); });

    size_type const nr = field_input.grid_r.size();
    size_type const nphi = field_input.grid_phi.size();
    size_type const nz = field_input.grid_z.size();
    size_type const total_points = nr * nphi * nz;

    // Prepare field data storage
    field_input.field.resize(static_cast<size_type>(CylAxis::size_)
                             * total_points);

    Array<size_type, 4> const dims{
        nr, nphi, nz, static_cast<size_type>(CylAxis::size_)};

    // Position calculator for cylindrical grid
    auto position_calculator = [&](size_type ir, size_type iphi, size_type iz) {
        auto r = r_grid[ir];
        auto phi = phi_values[iphi];
        auto z = z_grid[iz];
        return Array<G4double, 4>{r * std::cos(phi), r * std::sin(phi), z, 0};
    };

    // Field converter for cylindrical coordinates (requires coordinate
    // transformation)
    auto field_converter = [](Array<G4double, 3> const& bfield,
                              real_type* cur_bfield) {
        EnumArray<CylAxis, G4double> bfield_cyl;
        cartesian_to_cylindrical(bfield, bfield_cyl);
        auto bfield_cyl_native
            = convert_from_geant(bfield_cyl.data(), clhep_field);
        std::copy(
            bfield_cyl_native.cbegin(), bfield_cyl_native.cend(), cur_bfield);
    };

    // Sample field using common utility
    detail::setup_and_sample_field(
        field_input.field.data(), dims, position_calculator, field_converter);

    CELER_ENSURE(field_input);
    return field_input;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
