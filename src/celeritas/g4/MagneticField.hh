//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/g4/MagneticField.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <CLHEP/Units/SystemOfUnits.h>
#include <G4MagneticField.hh>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/math/Quantity.hh"
#include "geocel/g4/Convert.hh"
#include "celeritas/Quantities.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Wrap a Celeritas field for use in Geant4.
 * \tparam P params for creating field
 * \tparam F field calculator
 */
template<class P, class F>
class MagneticField : public G4MagneticField
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstParams = std::shared_ptr<P const>;
    //!@}

  public:
    // Construct with field params
    inline explicit MagneticField(SPConstParams params);

    // Calculate values of the magnetic field vector
    inline void
    GetFieldValue(G4double const point[3], G4double* field) const override;

  private:
    SPConstParams params_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with the Celeritas shared params.
 */
template<class P, class F>
MagneticField<P, F>::MagneticField(SPConstParams params)
    : params_(std::move(params))
{
    CELER_EXPECT(params_);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the magnetic field vector at the given position.
 */
template<class P, class F>
void MagneticField<P, F>::GetFieldValue(G4double const pos[3],
                                        G4double* field) const
{
    F calc_field(params_->host_ref());
    // Calculate the magnetic field value in the native Celeritas unit system
    Real3 result = calc_field(convert_from_geant(pos, clhep_length));
    for (auto i = 0; i < 3; ++i)
    {
        // Return values of the field vector in CLHEP::tesla for Geant4
        auto ft = native_value_to<units::FieldTesla>(result[i]);
        field[i] = convert_to_geant(ft.value(), CLHEP::tesla);
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
