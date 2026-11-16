//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file larceler/RunnerResults.cc
//---------------------------------------------------------------------------//
#include "RunnerResults.hh"

#include <iostream>
#include <larcoreobj/SimpleTypesAndConstants/geo_vectors.h>
#include <lardataobj/Simulation/OpDetBacktrackerRecord.h>
#include <lardataobj/Simulation/SimEnergyDeposit.h>

#include "corecel/cont/Range.hh"
#include "corecel/io/Repr.hh"
#include "celeritas/phys/GeneratorCounters.hh"

#include "AssertionHelper.hh"
#include "RefComparer.hh"
#include "TestMacros.hh"

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
    return os << '{' << CELER_REPR_ATTR(ref, num_hits) << '}';
}

bool is_ref_eq(
    AssertionHelper& helper, RunResult const& val1, RunResult const& val2)
{
    RefComparer<RunResult> compare{helper, val1, val2};
    CELER_INVOKE_TYPE_ATTR(compare.is_eq, RunResult, num_hits);
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
    return os << '{' << CELER_REPR_ATTR(ref, track_id) << ", "
              << CELER_REPR_ATTR(ref, num_photons) << ", "
              << CELER_REPR_ATTR(ref, energy) << ", "
              << CELER_REPR_ATTR(ref, pos) << '}';
}

//---------------------------------------------------------------------------//

bool is_ref_eq(AssertionHelper& helper,
               DepHitResult const& val1,
               DepHitResult const& val2)
{
    RefComparer<DepHitResult> compare{helper, val1, val2};
    CELER_INVOKE_TYPE_ATTR(compare.is_eq, DepHitResult, track_id);
    CELER_INVOKE_TYPE_ATTR(compare.is_soft_eq, DepHitResult, num_photons);
    CELER_INVOKE_TYPE_ATTR(compare.is_soft_eq, DepHitResult, energy);
    CELER_INVOKE_TYPE_ATTR(compare.is_soft_eq, DepHitResult, pos);
    return static_cast<bool>(compare);
}

//---------------------------------------------------------------------------//
std::ostream& operator<<(std::ostream& os, OpticalDistributionResult const& ref)
{
    return os << '{' << CELER_REPR_ATTR(ref, num_photons) << ", "
              << CELER_REPR_ATTR(ref, components) << ", "
              << CELER_REPR_ATTR(ref, primaries) << ",}";
}

bool is_ref_eq(AssertionHelper& helper,
               OpticalDistributionResult const& expected,
               OpticalDistributionResult const& actual)
{
    RefComparer<OpticalDistributionResult> compare{helper, expected, actual};
    CELER_INVOKE_TYPE_ATTR(
        compare.is_eq, OpticalDistributionResult, num_photons);
    CELER_INVOKE_TYPE_ATTR(
        compare.is_eq, OpticalDistributionResult, components);
    CELER_INVOKE_TYPE_ATTR(compare.is_eq, OpticalDistributionResult, primaries);
    return static_cast<bool>(compare);
}

//---------------------------------------------------------------------------//
void DiagnosticResult::append(CounterAccumStats const& stats)
{
    flushes.push_back(stats.flushes);
    num_cut.push_back(stats.num_cut);
    num_errored.push_back(stats.num_errored);
    step_iters.push_back(stats.step_iters);
    steps.push_back(stats.steps);
    gen_size.push_back(stats.generators.size());
    for (auto const& gen : stats.generators)
    {
        buffer_size.push_back(gen.buffer_size);
        num_generated.push_back(gen.num_generated);
        num_pending.push_back(gen.num_pending);
    }
}

std::ostream& operator<<(std::ostream& os, DiagnosticResult const& ref)
{
    return os << '{' << CELER_REPR_ATTR(ref, flushes) << ", "
              << CELER_REPR_ATTR(ref, num_cut) << ", "
              << CELER_REPR_ATTR(ref, num_errored) << ", "
              << CELER_REPR_ATTR(ref, step_iters) << ", "
              << CELER_REPR_ATTR(ref, steps) << ", "
              << CELER_REPR_ATTR(ref, gen_size) << ", "
              << CELER_REPR_ATTR(ref, buffer_size) << ", "
              << CELER_REPR_ATTR(ref, num_generated) << ", "
              << CELER_REPR_ATTR(ref, num_pending) << ",}";
}

bool is_ref_eq(AssertionHelper& helper,
               DiagnosticResult const& expected,
               DiagnosticResult const& actual)
{
    RefComparer<DiagnosticResult> compare{helper, expected, actual};
    CELER_INVOKE_TYPE_ATTR(compare.is_eq, DiagnosticResult, flushes);
    CELER_INVOKE_TYPE_ATTR(compare.is_eq, DiagnosticResult, num_cut);
    CELER_INVOKE_TYPE_ATTR(compare.is_eq, DiagnosticResult, num_errored);
    CELER_INVOKE_TYPE_ATTR(compare.is_eq, DiagnosticResult, step_iters);
    CELER_INVOKE_TYPE_ATTR(compare.is_eq, DiagnosticResult, steps);
    CELER_INVOKE_TYPE_ATTR(compare.is_eq, DiagnosticResult, gen_size);
    CELER_INVOKE_TYPE_ATTR(compare.is_eq, DiagnosticResult, buffer_size);
    CELER_INVOKE_TYPE_ATTR(compare.is_eq, DiagnosticResult, num_generated);
    CELER_INVOKE_TYPE_ATTR(compare.is_eq, DiagnosticResult, num_pending);
    return static_cast<bool>(compare);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
