//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/transform/Translation.test.cc
//---------------------------------------------------------------------------//
#include "orange/transform/Translation.hh"

#include <sstream>

#include "orange/transform/TransformIO.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
class TranslatorTest : public Test
{
};

TEST_F(TranslatorTest, output)
{
    Translation tr{Real3{1, 2, 3}};
    std::ostringstream os;
    os << tr;
    EXPECT_EQ("{{1,2,3}}", os.str());
}

TEST_F(TranslatorTest, translation)
{
    Translation tr(Real3{1, 2, 3});

    EXPECT_VEC_SOFT_EQ((Real3{.1, .2, .3}),
                       tr.transform_down(Real3{1.1, 2.2, 3.3}));
    EXPECT_VEC_SOFT_EQ((Real3{1.1, 2.2, 3.3}),
                       tr.transform_up(Real3{.1, .2, .3}));
}

TEST_F(TranslatorTest, rotation)
{
    Translation tr(Real3{1, 2, 3});

    Real3 dir{0, 0, 1};
    EXPECT_VEC_SOFT_EQ(dir, tr.rotate_down(dir));
    EXPECT_VEC_SOFT_EQ(dir, tr.rotate_up(dir));
}

TEST_F(TranslatorTest, serialization)
{
    Translation tr{Real3{3, 2, 1}};

    EXPECT_VEC_EQ((Real3{3, 2, 1}), tr.data());

    Translation tr2(tr.data());
    EXPECT_VEC_EQ((Real3{3, 2, 1}), tr2.translation());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
