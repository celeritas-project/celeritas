//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/Translator.test.cc
//---------------------------------------------------------------------------//
#include "orange/Translator.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
class TranslatorTest : public Test
{
  protected:
    Real3 translation_{1, 2, 3};
};

TEST_F(TranslatorTest, down)
{
    TranslatorDown translate(translation_);

    EXPECT_VEC_SOFT_EQ((Real3{.1, .2, .3}), translate(Real3{1.1, 2.2, 3.3}));
}

TEST_F(TranslatorTest, up)
{
    TranslatorUp translate(translation_);

    EXPECT_VEC_SOFT_EQ((Real3{1.1, 2.2, 3.3}), translate(Real3{.1, .2, .3}));
}

//---------------------------------------------------------------------------//
} // namespace test
} // namespace celeritas
