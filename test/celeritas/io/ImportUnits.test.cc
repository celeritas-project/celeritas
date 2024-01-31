//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportUnits.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/io/ImportUnits.hh"

#include "corecel/math/Algorithms.hh"
#include "celeritas/Units.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class ImportUnitsTest : public ::celeritas::test::Test
{
};

TEST_F(ImportUnitsTest, native)
{
    auto from_native = [](ImportUnits iu) {
        return native_value_from(UnitSystem::native, iu);
    };

    EXPECT_SOFT_EQ(1.0, from_native(ImportUnits::unitless));
    EXPECT_SOFT_EQ(1.0, from_native(ImportUnits::mev));
    EXPECT_SOFT_EQ(1.0, from_native(ImportUnits::mev_per_len));
    EXPECT_SOFT_EQ(1.0, from_native(ImportUnits::len));
    EXPECT_SOFT_EQ(1.0, from_native(ImportUnits::len_inv));
    EXPECT_SOFT_EQ(1.0, from_native(ImportUnits::len_mev_inv));
    EXPECT_SOFT_EQ(1.0, from_native(ImportUnits::mev_sq_per_len));
    EXPECT_SOFT_EQ(1.0, from_native(ImportUnits::len_sq));
    EXPECT_SOFT_EQ(1.0, from_native(ImportUnits::time));
    EXPECT_SOFT_EQ(1.0, from_native(ImportUnits::inv_len_cb));
}

TEST_F(ImportUnitsTest, csg)
{
    constexpr auto cm = units::centimeter;
    constexpr auto cm_sq = ipow<2>(units::centimeter);
    constexpr auto cm_cb = ipow<3>(units::centimeter);
    constexpr auto s = units::second;

    auto from_cgs = [](ImportUnits iu) {
        return native_value_from(UnitSystem::cgs, iu);
    };

    EXPECT_SOFT_EQ(1.0, from_cgs(ImportUnits::unitless));
    EXPECT_SOFT_EQ(1.0, from_cgs(ImportUnits::mev));
    EXPECT_SOFT_EQ(1.0 / cm, from_cgs(ImportUnits::mev_per_len));
    EXPECT_SOFT_EQ(cm, from_cgs(ImportUnits::len));
    EXPECT_SOFT_EQ(1 / cm, from_cgs(ImportUnits::len_inv));
    EXPECT_SOFT_EQ(1 / cm, from_cgs(ImportUnits::len_mev_inv));
    EXPECT_SOFT_EQ(1 / cm, from_cgs(ImportUnits::mev_sq_per_len));
    EXPECT_SOFT_EQ(cm_sq, from_cgs(ImportUnits::len_sq));
    EXPECT_SOFT_EQ(s, from_cgs(ImportUnits::time));
    EXPECT_SOFT_EQ(1 / cm_cb, from_cgs(ImportUnits::inv_len_cb));
}

TEST_F(ImportUnitsTest, clhep)
{
    constexpr auto mm = units::millimeter;
    constexpr auto mm_sq = ipow<2>(units::millimeter);
    constexpr auto mm_cb = ipow<3>(units::millimeter);
    constexpr auto ns = units::nanosecond;

    auto from_clhep = [](ImportUnits iu) {
        return native_value_from(UnitSystem::clhep, iu);
    };

    EXPECT_SOFT_EQ(1.0, from_clhep(ImportUnits::unitless));
    EXPECT_SOFT_EQ(1.0, from_clhep(ImportUnits::mev));
    EXPECT_SOFT_EQ(1.0 / mm, from_clhep(ImportUnits::mev_per_len));
    EXPECT_SOFT_EQ(mm, from_clhep(ImportUnits::len));
    EXPECT_SOFT_EQ(1 / mm, from_clhep(ImportUnits::len_inv));
    EXPECT_SOFT_EQ(1 / mm, from_clhep(ImportUnits::len_mev_inv));
    EXPECT_SOFT_EQ(1 / mm, from_clhep(ImportUnits::mev_sq_per_len));
    EXPECT_SOFT_EQ(mm_sq, from_clhep(ImportUnits::len_sq));
    EXPECT_SOFT_EQ(ns, from_clhep(ImportUnits::time));
    EXPECT_SOFT_EQ(1 / mm_cb, from_clhep(ImportUnits::inv_len_cb));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
