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
#include <G4FieldManager.hh>
#include <G4MagneticField.hh>
#include <G4TransportationManager.hh>
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

namespace celeritas
{
//---------------------------------------------------------------------------//
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
MakeCylMapFieldInput(std::vector<real_type> const& r_grid,
                     std::vector<real_type> const& phi_values,
                     std::vector<real_type> const& z_grid)
{
    CylMapFieldParams::Input field_input;
    field_input.grid_r.reserve(r_grid.size());
    field_input.grid_phi.reserve(phi_values.size());
    field_input.grid_z.reserve(z_grid.size());

    // Convert from geant
    std::transform(
        r_grid.cbegin(),
        r_grid.cend(),
        std::back_inserter(field_input.grid_r),
        [](real_type r) { return convert_from_geant(r, clhep_length); });
    //  Convert phi values to Turn type
    std::transform(phi_values.cbegin(),
                   phi_values.cend(),
                   std::back_inserter(field_input.grid_phi),
                   [](real_type phi) { return native_value_to<Turn>(phi); });
    std::transform(
        z_grid.cbegin(),
        z_grid.cend(),
        std::back_inserter(field_input.grid_z),
        [](real_type z) { return convert_from_geant(z, clhep_length); });

    size_type const nr = field_input.grid_r.size();
    size_type const nphi = field_input.grid_phi.size();
    size_type const nz = field_input.grid_z.size();
    size_type const total_points = nr * nphi * nz;

    field_input.field.resize(static_cast<size_type>(CylAxis::size_)
                             * total_points);

    Array<size_type, 4> const dims{
        nr, nphi, nz, static_cast<size_type>(CylAxis::size_)};
    HyperslabIndexer const flat_index{dims};

    CELER_EXPECT(G4TransportationManager::GetTransportationManager());
    CELER_EXPECT(
        G4TransportationManager::GetTransportationManager()->GetFieldManager());
    CELER_EXPECT(G4TransportationManager::GetTransportationManager()
                     ->GetFieldManager()
                     ->GetDetectorField());
    auto& field = *G4TransportationManager::GetTransportationManager()
                       ->GetFieldManager()
                       ->GetDetectorField();
    Array<G4double, 3> bfield;
    for (size_type ir = 0; ir < nr; ++ir)
    {
        real_type r = r_grid[ir];
        for (size_type iphi = 0; iphi < nphi; ++iphi)
        {
            real_type phi = phi_values[iphi];
            for (size_type iz = 0; iz < nz; ++iz)
            {
                auto* cur_bfield = field_input.field.data()
                                   + flat_index(ir, iphi, iz, 0);
                Array<G4double, 4> pos
                    = {r * std::cos(phi), r * std::sin(phi), z_grid[iz], 0};
                field.GetFieldValue(pos.data(), bfield.data());
                EnumArray<CylAxis, G4double> bfield_cyl;
                cartesian_to_cylindrical(bfield, bfield_cyl);
                auto bfield_cyl_native
                    = convert_from_geant(bfield_cyl.data(), clhep_field);
                std::copy(bfield_cyl_native.cbegin(),
                          bfield_cyl_native.cend(),
                          cur_bfield);
            }
        }
    }
    CELER_ENSURE(field_input);
    return field_input;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
