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
#include "RefComparer.hh"
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

std::ostream& operator<<(std::ostream& os, RunResult const& ref)
{
    return os << '{' << CELER_ATTR_REPR(ref, num_hits) << '}';
}

bool is_ref_eq(
    AssertionHelper& helper, RunResult const& val1, RunResult const& val2)
{
    RefComparer<RunResult> compare{helper, val1, val2};
    CELER_INVOKE_ATTR(compare.is_eq, RunResult, num_hits);
    return static_cast<bool>(compare);
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
        result.pos.push_back(Real3{sdp.x, sdp.y, sdp.z});
    }
    return result;
}

//---------------------------------------------------------------------------//

std::ostream& operator<<(std::ostream& os, DepHitResult const& ref)
{
    return os << '{' << CELER_ATTR_REPR(ref, track_id) << ", "
              << CELER_ATTR_REPR(ref, num_photons) << ", "
              << CELER_ATTR_REPR(ref, energy) << ", "
              << CELER_ATTR_REPR(ref, pos) << '}';
}

//---------------------------------------------------------------------------//

bool is_ref_eq(AssertionHelper& helper,
               DepHitResult const& val1,
               DepHitResult const& val2)
{
    RefComparer<DepHitResult> compare{helper, val1, val2};
    CELER_INVOKE_ATTR(compare.is_eq, DepHitResult, track_id);
    CELER_INVOKE_ATTR(compare.is_soft_eq, DepHitResult, num_photons);
    CELER_INVOKE_ATTR(compare.is_soft_eq, DepHitResult, energy);
    CELER_INVOKE_ATTR(compare.is_soft_eq, DepHitResult, pos);
    return static_cast<bool>(compare);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
