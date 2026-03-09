//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file larceler/PDFullSimCeler.cc
//---------------------------------------------------------------------------//
#include "PDFullSimCeler.hh"

#include <art/Framework/Principal/Event.h>
#include <art/Framework/Principal/Handle.h>
#include <larcore/CoreUtils/ServiceUtil.h>
#include <larcore/Geometry/Geometry.h>
#include <larcorealg/Geometry/OpDetGeo.h>
#include <lardataobj/Simulation/OpDetBacktrackerRecord.h>
#include <lardataobj/Simulation/SimEnergyDeposit.h>
#include <messagefacility/MessageLogger/MessageLogger.h>

#include "corecel/Assert.hh"

#include "LarStandaloneRunner.hh"
#include "larceler/Convert.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Convert from a FHiCL config input.
 */
inp::OpticalStandaloneInput
make_input_from_config(detail::PDFullSimCelerConfig const& cfg)
{
    inp::OpticalStandaloneInput result;

    // GPU options
    if (cfg.EnableDevice())
    {
        celeritas::inp::Device d;
        d.stack_size = cfg.DeviceStackSize();
        d.heap_size = cfg.DeviceHeapSize();
        result.system.device = d;
    }

    result.problem.generator = inp::OpticalOffloadGenerator{};

    // Optical limits
    if (auto steps = cfg.OpticalLimitSteps())
    {
        result.problem.limits.steps = steps;
    }
    if (auto step_iters = cfg.OpticalLimitStepIters())
    {
        result.problem.limits.step_iters = step_iters;
    }

    // Optical capacities
    result.problem.capacity.primaries = cfg.OpticalCapacityPrimaries();
    result.problem.capacity.tracks = cfg.OpticalCapacityTracks();
    result.problem.capacity.generators = cfg.OpticalCapacityGenerators();

    result.problem.num_streams = 1;
    result.problem.seed = cfg.Seed();
    result.problem.timers.action = cfg.ActionTimes();
    result.problem.output_file = cfg.OutputFile();

    return result;
}

}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with fcl parameters.
 */
PDFullSimCeler::PDFullSimCeler(Parameters const& config)
    : art::EDProducer{config}
    , runner_inp_{make_input_from_config(config())}
    , sim_tag_{config().SimulationLabel()}
{
    // Inform LArSoft we're going to make OpBTR
    produces<std::vector<sim::OpDetBacktrackerRecord>>();
}

//---------------------------------------------------------------------------//
/*!
 * Start Celeritas at the beginning of the job.
 */
void PDFullSimCeler::beginJob()
{
    CELER_EXPECT(!runner_);

    // Obtain the GDML filename from the LAr geometry service
    auto const* geo = lar::providerFrom<geo::Geometry>();
    CELER_ASSERT(geo);  // art checks this already
    runner_inp_.problem.model.geometry = geo->GDMLFile();

    // Set the special LArG4 detector name for optical detectors
    runner_inp_.detectors = {"PhotonDetector"};

    // Load optical detector positions
    std::vector<Real3> positions;
    for (auto i : range(geo->NOpDets()))
    {
        auto const& opdet = geo->OpDetGeoFromOpDet(i);
        positions.push_back(
            convert_from_larsoft<LarsoftLen>(opdet.GetCenter()));
    }

    runner_ = std::make_unique<LarStandaloneRunner>(
        std::forward<LarStandaloneRunner::Input>(runner_inp_),
        std::move(positions));
}

//---------------------------------------------------------------------------//
/*!
 * Run Celeritas on a single event.
 */
void PDFullSimCeler::produce(art::Event& e)
{
    CELER_EXPECT(runner_);
    auto edep_handle
        = e.getValidHandle<std::vector<sim::SimEnergyDeposit>>(sim_tag_);

    // Calculate detector response for the input steps
    using VecBTR = LarStandaloneRunner::VecBTR;
    VecBTR result = (*runner_)(*edep_handle);

    // Add to event
    e.put(std::make_unique<VecBTR>(std::move(result)));
}

//---------------------------------------------------------------------------//
/*!
 * Free Celeritas memory at the end of the job.
 */
void PDFullSimCeler::endJob()
{
    runner_.reset();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
