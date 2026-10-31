//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file larceler/RunnerResults.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include <iosfwd>
#include <vector>
#include <gtest/gtest.h>

#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"  // IWYU pragma: keep

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
    static constexpr char class_str[] = "RunResult";

    std::vector<int> num_hits;

    static RunResult from_btr(
        std::vector<sim::OpDetBacktrackerRecord> const& records);

    friend std::ostream& operator<<(std::ostream&, RunResult const&);
};

//---------------------------------------------------------------------------//

bool is_ref_eq(AssertionHelper& helper,
               RunResult const& expected,
               RunResult const& actual);

//---------------------------------------------------------------------------//
struct DepHitResult
{
    static constexpr char class_str[] = "DepHitResult";

    using Real3 = Array<float, 3>;

    std::vector<int> track_id;
    std::vector<float> num_photons;
    std::vector<float> energy;
    std::vector<Real3> pos;

    static DepHitResult from_sdp(std::vector<sim::SDP> const& sdps);

    friend std::ostream& operator<<(std::ostream&, DepHitResult const&);
};

//---------------------------------------------------------------------------//

bool is_ref_eq(AssertionHelper& helper,
               DepHitResult const& expected,
               DepHitResult const& actual);

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
