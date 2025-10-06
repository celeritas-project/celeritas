//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/g4org/Transformer.test.cc
//---------------------------------------------------------------------------//
#include "orange/g4org/Transformer.hh"

#include <G4AffineTransform.hh>
#include <G4RotationMatrix.hh>
#include <G4ThreeVector.hh>

#include "corecel/cont/ArrayIO.hh"
#include "corecel/math/ArrayOperators.hh"
#include "corecel/math/ArrayUtils.hh"
#include "geocel/UnitUtils.hh"
#include "geocel/detail/LengthUnits.hh"
#include "geocel/g4/Convert.hh"
#include "orange/MatrixUtils.hh"
#include "orange/g4org/Scaler.hh"
#include "orange/transform/TransformIO.hh"

#include "celeritas_test.hh"

using celeritas::test::to_cm;

namespace celeritas
{
namespace g4org
{
namespace test
{
//---------------------------------------------------------------------------//
constexpr real_type mm{::celeritas::lengthunits::millimeter};

Real3 from_geant(G4ThreeVector const& tv)
{
    return convert_from_geant(tv);
}

G4ThreeVector to_geant(Real3 const& rv)
{
    return {rv[0], rv[1], rv[2]};
}

class TransformerTest : public ::celeritas::test::Test
{
  protected:
    Scaler scale_;
};

TEST_F(TransformerTest, translation)
{
    Transformer transform{scale_};

    auto trans = transform(G4ThreeVector(1.0, 2.0, 3.0));
    EXPECT_VEC_SOFT_EQ((Real3{1 * mm, 2 * mm, 3 * mm}), trans.translation());
}

TEST_F(TransformerTest, affine_transform)
{
    // Daughter to parent: +x becomes +y
    Real3 const rot_axis = make_unit_vector(Real3{3, 4, 5});
    Turn const rot_turn{0.125};

    // Construct Geant4 matrix and transforms
    G4RotationMatrix g4mat(to_geant(rot_axis), native_value_from(rot_turn));
    G4AffineTransform g4tran{g4mat, G4ThreeVector(10.0, 20.0, 30.0)};

    // Parent-to-daughter
    auto g4tran_inv = g4tran.Inverse();

    // Construct Celeritas matrix
    auto const mat = make_rotation(rot_axis, rot_turn);
    {
        auto actual = convert_from_geant(g4mat);
        // Check raw matrix conversion
        EXPECT_VEC_SOFT_EQ(mat[0], actual[0]);
        EXPECT_VEC_SOFT_EQ(mat[1], actual[1]);
        EXPECT_VEC_SOFT_EQ(mat[2], actual[2]);
    }
    EXPECT_VEC_SOFT_EQ(gemv(mat, Real3{1, 0, 0}),
                       from_geant(g4mat(G4ThreeVector(1, 0, 0))));

    // Expected Celeritas transform
    Transformation const tran(make_transpose(mat),
                              Real3{10 * mm, 20 * mm, 30 * mm});
    EXPECT_VEC_SOFT_EQ(tran.translation(), scale_(g4tran.NetTranslation()));

    Transformer transform{scale_};
    {
        Transformation const actual = transform(g4tran);
        EXPECT_VEC_SOFT_EQ(tran.data(), actual.data());

        EXPECT_VEC_SOFT_EQ(
            from_geant(g4tran.TransformAxis(G4ThreeVector{1, 0, 0})),
            actual.rotate_up(Real3{1, 0, 0}));
        EXPECT_VEC_SOFT_EQ(
            from_geant(g4tran.TransformAxis(G4ThreeVector{0, 1, 0})),
            actual.rotate_up(Real3{0, 1, 0}));
        EXPECT_VEC_SOFT_EQ(
            from_geant(g4tran.TransformAxis(G4ThreeVector{0, 0, 1})),
            actual.rotate_up(Real3{0, 0, 1}));

        // Do local-to-global transform from mm.
        // See G4DisplacedSolid::DistanceToIn
        G4ThreeVector const g4daughter{100, 200, 300};

        EXPECT_VEC_SOFT_EQ(scale_(g4tran.TransformPoint(g4daughter)),
                           actual.transform_up(scale_(g4daughter)));
        EXPECT_VEC_SOFT_EQ(scale_(g4tran_inv.TransformPoint(g4daughter)),
                           actual.transform_down(scale_(g4daughter)));
        EXPECT_VEC_SOFT_EQ(
            from_geant(g4tran_inv.TransformAxis(G4ThreeVector{1, 0, 0})),
            actual.rotate_down(Real3{1, 0, 0}));
        EXPECT_VEC_SOFT_EQ(
            from_geant(g4tran_inv.TransformAxis(G4ThreeVector{0, 1, 0})),
            actual.rotate_down(Real3{0, 1, 0}));
        EXPECT_VEC_SOFT_EQ(
            from_geant(g4tran_inv.TransformAxis(G4ThreeVector{0, 0, 1})),
            actual.rotate_down(Real3{0, 0, 1}));
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace g4org
}  // namespace celeritas
