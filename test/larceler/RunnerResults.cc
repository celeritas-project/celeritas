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

DepHitResult DepHitResult::from_sdp(std::vector<sim::SDP> const& sdps)
{
    DepHitResult result;
    for (auto const& sdp : sdps)
    {
        result.track_id.push_back(sdp.trackID);
        result.num_photons.push_back(sdp.numPhotons);
        result.energy.push_back(sdp.energy);
        result.x.push_back(sdp.x);
        result.y.push_back(sdp.y);
        result.z.push_back(sdp.z);
    }
    return result;
}

//---------------------------------------------------------------------------//

void DepHitResult::print_expected() const
{
    std::cout << "DepHitResult ref;\n"
              << "ref.track_id = " << repr(track_id) << ";\n"
              << "ref.num_photons = " << repr(num_photons) << ";\n"
              << "ref.energy = " << repr(energy) << ";\n"
              << "ref.x = " << repr(x) << ";\n"
              << "ref.y = " << repr(y) << ";\n"
              << "ref.z = " << repr(z) << ";\n"
              << "EXPECT_REF_EQ(ref, result);\n";
}

//---------------------------------------------------------------------------//

::testing::AssertionResult IsRefEq(char const* expr1,
                                   char const* expr2,
                                   DepHitResult const& val1,
                                   DepHitResult const& val2)
{
    AssertionHelper helper{expr1, expr2};

    if (auto r = ::celeritas::testdetail::IsVecEq(
            expr1, "track_id", val1.track_id, val2.track_id);
        !static_cast<bool>(r))
    {
        helper.fail() << r.message();
    }

    if (helper.equal_size(val1.num_photons.size(), val2.num_photons.size()))
    {
        auto softeq_result = ::celeritas::testdetail::IsVecSoftEquiv(
            expr1, expr2, val1.num_photons, val2.num_photons);
        if (!softeq_result)
        {
            helper.fail() << "  num_photons: " << softeq_result.message();
        }
    }
    else
    {
        helper.fail() << "  num_photons: " << expr2 << " = "
                      << repr(val2.num_photons);
    }

    if (helper.equal_size(val1.energy.size(), val2.energy.size()))
    {
        auto softeq_result = ::celeritas::testdetail::IsVecSoftEquiv(
            expr1, expr2, val1.energy, val2.energy);
        if (!softeq_result)
        {
            helper.fail() << "  energy: " << softeq_result.message();
        }
    }
    else
    {
        helper.fail() << "  energy: " << expr2 << " = " << repr(val2.energy);
    }

    if (helper.equal_size(val1.x.size(), val2.x.size()))
    {
        auto softeq_result = ::celeritas::testdetail::IsVecSoftEquiv(
            expr1, expr2, val1.x, val2.x);
        if (!softeq_result)
        {
            helper.fail() << "  x: " << softeq_result.message();
        }
    }
    else
    {
        helper.fail() << "  x: " << expr2 << " = " << repr(val2.x);
    }

    if (helper.equal_size(val1.y.size(), val2.y.size()))
    {
        auto softeq_result = ::celeritas::testdetail::IsVecSoftEquiv(
            expr1, expr2, val1.y, val2.y);
        if (!softeq_result)
        {
            helper.fail() << "  y: " << softeq_result.message();
        }
    }
    else
    {
        helper.fail() << "  y: " << expr2 << " = " << repr(val2.y);
    }

    if (helper.equal_size(val1.z.size(), val2.z.size()))
    {
        auto softeq_result = ::celeritas::testdetail::IsVecSoftEquiv(
            expr1, expr2, val1.z, val2.z);
        if (!softeq_result)
        {
            helper.fail() << "  z: " << softeq_result.message();
        }
    }
    else
    {
        helper.fail() << "  z: " << expr2 << " = " << repr(val2.z);
    }

    return helper;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
