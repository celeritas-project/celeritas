//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/PrimaryCapacity.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/global/PrimaryCapacity.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
TEST(PrimaryCapacityTest, vacancy_and_secondary_reserve)
{
    detail::PrimaryCapacityInput input;
    input.track_slots = 8;
    input.initializer_capacity = 128;
    input.secondary_capacity = 120;

    // Vacancies consume eight primaries before the remaining eight use the
    // initializer space that is not reserved for secondaries.
    EXPECT_EQ(16, detail::calc_primary_capacity(input));

    input.alive = 8;
    EXPECT_EQ(8, detail::calc_primary_capacity(input));

    input.alive = 4;
    input.queued = 8;
    EXPECT_EQ(4, detail::calc_primary_capacity(input));

    input.alive = 0;
    EXPECT_EQ(8, detail::calc_primary_capacity(input));

    input.queued = input.initializer_capacity;
    EXPECT_EQ(0, detail::calc_primary_capacity(input));
}

//---------------------------------------------------------------------------//
TEST(PrimaryCapacityTest, oversized_secondary_stack)
{
    detail::PrimaryCapacityInput input;
    input.track_slots = 1;
    input.initializer_capacity = 1;
    input.secondary_capacity = 2;

    // A primary consumed into a vacant slot does not reduce the space
    // available for secondaries.
    EXPECT_EQ(1, detail::calc_primary_capacity(input));

    input.alive = 1;
    EXPECT_EQ(0, detail::calc_primary_capacity(input));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
