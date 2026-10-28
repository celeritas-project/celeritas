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

RunResult RunResult::from_btr(
    std::vector<sim::OpDetBacktrackerRecord> const& response)
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

void check_ref_eq(AssertionHelper& helper,
                  RunResult const& val1,
                  RunResult const& val2,
                  std::nullptr_t)
{
    is_vec_eq(helper, "num_hits", val1.num_hits, val2.num_hits);
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

void check_ref_eq(AssertionHelper& helper,
                  DepHitResult const& val1,
                  DepHitResult const& val2,
                  std::nullptr_t)
{
    is_eq(helper, "track_id", val1.track_id, val2.track_id);
    is_vec_soft_eq(helper, "num_photons", val1.num_photons, val2.num_photons);
    is_vec_soft_eq(helper, "energy", val1.energy, val2.energy);
    is_vec_soft_eq(helper, "x", val1.x, val2.x);
    is_vec_soft_eq(helper, "y", val1.y, val2.y);
    is_vec_soft_eq(helper, "z", val1.z, val2.z);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
