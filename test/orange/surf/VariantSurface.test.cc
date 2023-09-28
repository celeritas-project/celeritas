//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/VariantSurface.test.cc
//---------------------------------------------------------------------------//
#include "orange/surf/VariantSurface.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class VariantSurfaceTest : public ::celeritas::test::Test
{
  protected:
    void SetUp() override {}
};

TEST_F(VariantSurfaceTest, apply_translation)
{
    auto result = apply_transform(Translation{Real3{0, 1, 0}}, PlaneY{4});
    if (auto* py = std::get_if<PlaneY>(&result))
    {
        EXPECT_SOFT_EQ(5, py->position());
    }
    else
    {
        FAIL() << "wrong type";
    }
}

TEST_F(VariantSurfaceTest, apply_transformation)
{
    auto result = apply_transform(
        Transformation{make_rotation(Axis::z, Turn{0.25}), Real3{1, 2, 0}},
        PlaneX{1});

    if (auto* p = std::get_if<Plane>(&result))
    {
        EXPECT_VEC_SOFT_EQ((Real3{0, 1, 0}), p->normal());
        EXPECT_SOFT_EQ(3, p->displacement());
    }
    else
    {
        FAIL() << "wrong type";
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
