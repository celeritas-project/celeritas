//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/g4/MagneticField.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/g4/MagneticField.hh"

#include <CLHEP/Units/SystemOfUnits.h>

#include "celeritas/Quantities.hh"
#include "celeritas/field/UniformField.hh"
#include "celeritas/field/UniformFieldParams.hh"
#include "celeritas/inp/Field.hh"
#include "celeritas/io/ImportUnits.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
struct TestFieldData
{
    real_type strength{1};  // [native Bfield]
    real_type z_offset{0};  // [native len]
};

class TestFieldParams
{
  public:
    explicit TestFieldParams(TestFieldData const& d) : data_{d} {}

    TestFieldData const& host_ref() const { return data_; }

  private:
    TestFieldData data_;
};

//! Return strength if greater than offset, otherwise zero
struct TestField
{
    TestFieldData const& data;

    Real3 operator()(Real3 const& pos) const
    {
        if (pos[2] > data.z_offset)
            return {0, 0, data.strength};
        return {0, 0, 0};
    }
};

//---------------------------------------------------------------------------//

class MagneticFieldTest : public ::celeritas::test::Test
{
};

TEST_F(MagneticFieldTest, uniform)
{
    auto params = std::make_shared<UniformFieldParams>([] {
        inp::UniformField f;
        f.strength = {0.5, 0.3, 1.0};
        EXPECT_EQ(UnitSystem::si, f.units);
        return f;
    }());

    MagneticField<UniformFieldParams, UniformField> g4_field(params);

    G4double pos[3] = {0, 0, 0};
    G4double field[3] = {0, 0, 0};

    g4_field.GetFieldValue(pos, field);

    // NOTE: quantities are cast to single precision
    EXPECT_SOFT_EQ(real_type(0.5 * CLHEP::tesla), field[0]);
    EXPECT_SOFT_EQ(real_type(0.3 * CLHEP::tesla), field[1]);
    EXPECT_SOFT_EQ(real_type(1.0 * CLHEP::tesla), field[2]);
}

TEST_F(MagneticFieldTest, nonuniform)
{
    auto params = std::make_shared<TestFieldParams>([] {
        TestFieldData d;
        d.strength = native_value_from(units::FieldTesla{1.5});
        d.z_offset = native_value_from(units::CmLength{0.7});
        return d;
    }());

    MagneticField<TestFieldParams, TestField> g4_field(params);

    using Dbl3 = Array<double, 3>;
    Array<double, 4> xyzt{0, 0, 0, 0};
    Dbl3 field{-1, -1, -1};

    g4_field.GetFieldValue(xyzt.data(), field.data());
    EXPECT_VEC_SOFT_EQ((Dbl3{0, 0, 0}), field);

    xyzt = {0, 0, 0.71 * CLHEP::cm, 0};
    g4_field.GetFieldValue(xyzt.data(), field.data());
    EXPECT_VEC_SOFT_EQ((Dbl3{0, 0, 1.5 * CLHEP::tesla}), field);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
