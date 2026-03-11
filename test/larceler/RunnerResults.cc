//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file larceler/RunnerResults.cc
//---------------------------------------------------------------------------//
#include "RunnerResults.hh"

#include <larcoreobj/SimpleTypesAndConstants/geo_vectors.h>
#include <lardataobj/Simulation/OpDetBacktrackerRecord.h>
#include <lardataobj/Simulation/SimEnergyDeposit.h>

#include "corecel/cont/Range.hh"
#include "corecel/io/Repr.hh"

#include "AssertionHelper.hh"
#include "testdetail/TestMacrosImpl.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

RunResult
RunResult::from_btr(std::vector<sim::OpDetBacktrackerRecord> const& response)
{
    RunResult result;
    for (auto i : range(response.size()))
    {
        // Shouldn't have hits on top detector
        auto const& btr = response[i];
        EXPECT_EQ(i, btr.OpDetNum());
        auto const& hits = btr.timePDclockSDPsMap();
        result.num_hits.push_back(hits.size());
    }
    return result;
}

//---------------------------------------------------------------------------//

void RunResult::print_expected() const
{
    std::cout << "RunResult ref;\n"
              << "ref.num_hits = " << repr(num_hits) << ";\n"
              << "EXPECT_REF_EQ(ref, result);\n";
}

::testing::AssertionResult IsRefEq(char const* expr1,
                                   char const* expr2,
                                   RunResult const& val1,
                                   RunResult const& val2)
{
    AssertionHelper helper{expr1, expr2};

    if (auto r = ::celeritas::testdetail::IsVecEq(
            expr1, "num_hits", val1.num_hits, val2.num_hits);
        !static_cast<bool>(r))
    {
        helper.fail() << r.message();
    }

    return helper;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
