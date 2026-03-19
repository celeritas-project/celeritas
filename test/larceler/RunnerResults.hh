//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file larceler/RunnerResults.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>
#include <gtest/gtest.h>

namespace sim
{
class OpDetBacktrackerRecord;
class OBTRHelper;
struct SDP;
}  // namespace sim

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
struct RunResult
{
    std::vector<int> num_hits;

    static RunResult
    from_btr(std::vector<sim::OpDetBacktrackerRecord> const& records);

    void print_expected() const;
};

//---------------------------------------------------------------------------//

::testing::AssertionResult IsRefEq(char const* expr1,
                                   char const* expr2,
                                   RunResult const& val1,
                                   RunResult const& val2);

//---------------------------------------------------------------------------//
struct DepHitResult
{
    std::vector<int> track_id;
    std::vector<float> num_photons;
    std::vector<float> energy;
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> z;

    static DepHitResult from_sdp(std::vector<sim::SDP> const& sdps);

    void print_expected() const;
};

//---------------------------------------------------------------------------//

::testing::AssertionResult IsRefEq(char const* expr1,
                                   char const* expr2,
                                   DepHitResult const& val1,
                                   DepHitResult const& val2);

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
