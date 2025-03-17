//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/RZPhiMapMagneticField.cc
//---------------------------------------------------------------------------//

#include "RZPhiMapMagneticField.hh"

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
#include "corecel/io/Logger.hh"
#include "corecel/math/Quantity.hh"
#include "corecel/math/Turn.hh"
#include "geocel/g4/Convert.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/ext/GeantUnits.hh"
#include "celeritas/field/RZPhiMapFieldInput.hh"
#include "celeritas/field/RZPhiMapFieldParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
namespace
{
//---------------------------------------------------------------------------//

//! Cartesion to cylindrical 3D vector conversion, cylindrical vector is
//! ordered as Phi-R-Z
inline void cartesian_to_cylindrical(Array<G4double, 3> const& cart,
                                     EnumArray<CylAxis, real_type>& cyl)
{
    double const phi = std::atan2(cart[1], cart[0]);
    cyl[CylAxis::Phi] = -cart[0] * std::sin(phi) + cart[1] * std::cos(phi);
    cyl[CylAxis::R] = cart[0] * std::cos(phi) + cart[1] * std::sin(phi);
    cyl[CylAxis::Z] = cart[2];
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Generates input for RZPhiMapField params with configurable nonuniform grid
 * dimensions in native Geant4 units, and \f$\phi\f$ should be in the range
 * [0;\f$2\times\pi\f$]. This must be called after G4RunManager::Initialize as
 * it will retrieve the G4FieldManager's field to sample it.
 */
RZPhiMapFieldParams::Input
MakeRZPhiMapFieldInput(std::vector<real_type> const& r_grid,
                       std::vector<real_type> const& z_grid,
                       std::vector<real_type> const& phi_values)
{
    RZPhiMapFieldParams::Input field_input;
    field_input.grid_r.reserve(r_grid.size());
    field_input.grid_z.reserve(z_grid.size());
    field_input.grid_phi.reserve(phi_values.size());

    // Convert from geant
    std::transform(
        r_grid.cbegin(),
        r_grid.cend(),
        std::back_inserter(field_input.grid_r),
        [](real_type r) { return convert_from_geant(r, clhep_length); });
    std::transform(
        z_grid.cbegin(),
        z_grid.cend(),
        std::back_inserter(field_input.grid_z),
        [](real_type z) { return convert_from_geant(z, clhep_length); });

    //  Convert phi values to Turn type
    std::transform(phi_values.cbegin(),
                   phi_values.cend(),
                   std::back_inserter(field_input.grid_phi),
                   [](real_type phi) { return native_value_to<Turn>(phi); });

    size_type const nr = field_input.grid_r.size();
    size_type const nz = field_input.grid_z.size();
    size_type const nphi = field_input.grid_phi.size();
    size_type const total_points = nr * nz * nphi;

    field_input.field.resize(static_cast<size_type>(CylAxis::size_)
                             * total_points);

    Array<size_type, 4> const dims{
        nphi, nr, nz, static_cast<size_type>(CylAxis::size_)};
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
    for (size_type iphi = 0; iphi < nphi; ++iphi)
    {
        real_type phi = native_value_from(field_input.grid_phi[iphi]);
        for (size_type ir = 0; ir < nr; ++ir)
        {
            real_type r = field_input.grid_r[ir];
            for (size_type iz = 0; iz < nz; ++iz)
            {
                auto idx = flat_index(iphi, ir, iz, 0);
                Array<G4double, 4> pos = {r * std::cos(phi),
                                          r * std::sin(phi),
                                          field_input.grid_z[iz],
                                          0};
                field.GetFieldValue(pos.data(), bfield.data());
                EnumArray<CylAxis, real_type> bfield_cyl;
                cartesian_to_cylindrical(bfield, bfield_cyl);
                auto bfield_cyl_g4
                    = convert_from_geant(bfield_cyl.data(), clhep_field);
                for (auto comp : range(CylAxis::size_))
                {
                    field_input.field[idx + static_cast<size_type>(comp)]
                        = bfield_cyl_g4[static_cast<size_type>(comp)];
                }
                CELER_LOG(info) << "Field at r=" << r << " cm, phi=" << phi
                                << " rad, z=" << field_input.grid_z[iz]
                                << " cm: " << field_input.field[idx] << " T, "
                                << field_input.field[idx + 1] << " T, "
                                << field_input.field[idx + 2] << " T";
            }
        }
    }
    CELER_ENSURE(field_input);
    return field_input;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
