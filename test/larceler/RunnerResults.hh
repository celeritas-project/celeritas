//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file larceler/RunnerResults.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
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
class AssertionHelper;
//---------------------------------------------------------------------------//
struct RunResult
{
    std::vector<int> num_hits;

    static RunResult from_btr(
        std::vector<sim::OpDetBacktrackerRecord> const& records);

    void print_expected() const;
};

//---------------------------------------------------------------------------//

void check_ref_eq(AssertionHelper& helper,
                  RunResult const& expected,
                  RunResult const& actual,
                  std::nullptr_t);

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

void check_ref_eq(AssertionHelper& helper,
                  DepHitResult const& expected,
                  DepHitResult const& actual,
                  std::nullptr_t);

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
