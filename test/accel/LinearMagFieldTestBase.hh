//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/LinearMagFieldTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <CLHEP/Units/SystemOfUnits.h>
#include <G4MagneticField.hh>

#include "corecel/Types.hh"
#include "corecel/cont/ArrayIO.hh"  // IWYU pragma: keep
#include "corecel/math/ArrayOperators.hh"
#include "geocel/UnitUtils.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/g4/MagneticField.hh"

#include "Test.hh"
#include "TestMacros.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Data for a linear magnetic field.
 *
 * The field is zero at the origin and increases linearly with distance. All
 * units are in the native system.
 */
struct LinearMagFieldData
{
    real_type scale{1};  //!< Field scale factor [Bfield / len]
    Real3 origin{0};  //!< Origin point where field is zero [len]
};

//---------------------------------------------------------------------------//
/*!
 * Parameters for the test linear magnetic field.
 */
class LinearMagFieldParams
{
  public:
    explicit LinearMagFieldParams(LinearMagFieldData const& d) : data_{d} {}

    LinearMagFieldData const& host_ref() const { return data_; }

  private:
    LinearMagFieldData data_;
};

//---------------------------------------------------------------------------//
/*!
 * Linear magnetic field functor.
 *
 * Returns a field value that is zero at origin and increases linearly:
 * \f[
 *   \vec{B}(\vec{r}) = s \cdot (\vec{r} - \vec{r}_0)
 * \f]
 * where \em s is the scale factor and \f$ \vec{r}_0 \f$ is the origin.
 */
struct LinearMagField
{
    LinearMagFieldData const& data;

    //! Calculate field at the given position
    Real3 operator()(Real3 const& pos) const
    {
        return data.scale * (pos - data.origin);
    }
};

//---------------------------------------------------------------------------//
// Test harness that creates a Geant4 magnetic field
class LinearMagFieldTestBase : public ::celeritas::test::Test
{
  public:
    using MagFieldT = MagneticField<LinearMagFieldParams, LinearMagField>;
    using Dbl3 = Array<double, 3>;

    void SetUp()
    {
        auto params = std::make_shared<LinearMagFieldParams>([] {
            LinearMagFieldData d;
            d.scale = native_value_from(units::FieldTesla{1.5})
                      / native_value_from(units::CmLength{1});
            d.origin = from_cm({0.7, 1.1, -2.5});
            return d;
        }());

        g4field_ = std::make_unique<MagFieldT>(params);

        // Check where value should be {0,0,0} and {0,0,1.5}
        using CLHEP::cm;
        EXPECT_VEC_NEAR(
            (Dbl3{0, 0, 0}),
            calc_field(this->g4field(), {0.7 * cm, 1.1 * cm, -2.5 * cm}),
            1e-6);
        EXPECT_VEC_NEAR(
            (Dbl3{0, 0, 1.5}),
            calc_field(this->g4field(), {0.7 * cm, 1.1 * cm, -1.5 * cm}),
            1e-6);
    }

    G4MagneticField const& g4field() const { return *g4field_; }

    static Dbl3 calc_field(G4MagneticField const& field, Dbl3 const& pos)
    {
        Array<double, 4> xyzt{pos[0], pos[1], pos[2], 0};
        Dbl3 result{-1, -1, -1};
        field.GetFieldValue(xyzt.data(), result.data());
        result /= CLHEP::tesla;
        return result;
    }

    template<class T>
    void
    check_field(G4MagneticField const& actual, Dbl3 const& pos, T cmp) const
    {
        EXPECT_VEC_NEAR(
            calc_field(this->g4field(), pos), calc_field(actual, pos), cmp)
            << "at " << pos << " [mm]";
    }

  protected:
    std::unique_ptr<MagFieldT> g4field_;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
