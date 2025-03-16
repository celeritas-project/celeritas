//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/RZPhiMapMagneticField.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>
#include <memory>
#include <CLHEP/Units/SystemOfUnits.h>
#include <G4FieldManager.hh>
#include <G4MagneticField.hh>
#include <G4TransportationManager.hh>
#include <corecel/Assert.hh>

#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/math/Quantity.hh"
#include "corecel/math/Turn.hh"
#include "geocel/g4/Convert.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/field/RZPhiMapField.hh"
#include "celeritas/field/RZPhiMapFieldInput.hh"
#include "celeritas/field/RZPhiMapFieldParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Generates input for RZPhiMapField params with configurable nonuniform grid
 * dimensions in native Geant4 units. This must be called after
 * G4RunManager::Initialize as it will retrieve the G4FieldManager's field to
 * sample it.
 */
class RZPhiMapFieldSampler
{
  public:
    // Generate field with user-defined grid
    RZPhiMapFieldParams::Input
    operator()(std::vector<real_type> r_grid,
               std::vector<real_type> z_grid,
               std::vector<real_type> phi_values) const
    {
        RZPhiMapFieldParams::Input field_input;

        field_input.grid_r = std::move(r_grid);
        field_input.grid_z = std::move(z_grid);

        // Convert phi values to Turn type
        field_input.grid_phi.resize(phi_values.size());
        std::transform(
            phi_values.cbegin(),
            phi_values.cend(),
            field_input.grid_phi.begin(),
            [](real_type phi) { return native_value_to<Turn>(phi); });

        size_type const nr = field_input.grid_r.size();
        size_type const nz = field_input.grid_z.size();
        size_type const nphi = field_input.grid_phi.size();
        size_type const total_points = nr * nz * nphi;

        field_input.field_r.resize(total_points);
        field_input.field_phi.resize(total_points);
        field_input.field_z.resize(total_points);

        Array<size_type, 3> const dims{nphi, nr, nz};
        HyperslabIndexer const flat_index{dims};
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
                    auto idx = flat_index(iphi, ir, iz);
                    Array<G4double, 4> pos = {r * std::cos(phi),
                                              r * std::sin(phi),
                                              field_input.grid_z[iz],
                                              0};
                    field.GetFieldValue(pos.data(), bfield.data());

                    // values in cylindrical vector space
                    field_input.field_r[idx] = bfield[0] * std::cos(phi)
                                               + bfield[1] * std::sin(phi);
                    field_input.field_phi[idx] = -bfield[0] * std::sin(phi)
                                                 + bfield[1] * std::cos(phi);
                    field_input.field_z[idx] = bfield[2];
                }
            }
        }
        return field_input;
    }
};

//---------------------------------------------------------------------------//
/*!
 * A user magnetic field equivalent to celeritas::RZPhiMapField.
 */
class RZPhiMapMagneticField : public G4MagneticField
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstFieldParams = std::shared_ptr<RZPhiMapFieldParams const>;
    //!@}

  public:
    // Construct with RZPhiMapFieldParams
    inline explicit RZPhiMapMagneticField(SPConstFieldParams field_params);

    // Calculate values of the magnetic field vector
    inline void
    GetFieldValue(double const point[3], double* field) const override;

  private:
    SPConstFieldParams params_;
    RZPhiMapField calc_field_;
};

//---------------------------------------------------------------------------//
/*!
 * Construct with the Celeritas shared RZPhiMapFieldParams.
 */
RZPhiMapMagneticField::RZPhiMapMagneticField(SPConstFieldParams params)
    : params_(std::move(params))
    , calc_field_(RZPhiMapField{params_->ref<MemSpace::native>()})
{
    CELER_EXPECT(params_);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the magnetic field vector at the given position.
 */
void RZPhiMapMagneticField::GetFieldValue(double const pos[3],
                                          double* field) const
{
    // Calculate the magnetic field value in the native Celeritas unit system
    Real3 result = calc_field_(convert_from_geant(pos, clhep_length));
    for (auto i = 0; i < 3; ++i)
    {
        // Return values of the field vector in CLHEP::tesla for Geant4
        auto ft = native_value_to<units::FieldTesla>(result[i]);
        field[i] = convert_to_geant(ft.value(), CLHEP::tesla);
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
