//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file larceler/LarCelerStandalone.cc
//---------------------------------------------------------------------------//
#include "LarCelerStandalone.hh"

#include <memory>
#include <larcore/CoreUtils/ServiceUtil.h>
#include <larcore/Geometry/Geometry.h>
#include <lardataobj/Simulation/OpDetBacktrackerRecord.h>
#include <lardataobj/Simulation/SimEnergyDeposit.h>

#include "corecel/Assert.hh"
#include "celeritas/inp/StandaloneInput.hh"

#include "LarStandaloneRunner.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Convert from a FHiCL config input.
 */
inp::OpticalStandaloneInput
make_input_from_config(detail::LarCelerStandaloneConfig const& cfg)
{
    inp::OpticalStandaloneInput result;

#if 0
    // FIXME: environment config doesn't yet work
    {
        fhicl::ParameterSet const& ps = cfg.environment();
        for (auto const& key : ps.get_names()) {
            result.environment[key] = ps.get<std::string>(key);
        }
    }
#endif

    // GPU options
    if (cfg.device().enable())
    {
        celeritas::inp::Device d;
        d.stack_size = cfg.device().stack_size();
        d.heap_size = cfg.device().heap_size();
        result.system.device = d;
    }

    // Obtain the GDML filename from the LAr geometry service
    auto geo_handle = lar::providerFrom<geo::Geometry>();
    CELER_VALIDATE(geo_handle, << "LArSoft geometry is not active");

    result.problem.model.geometry = geo_handle->GDMLFile();
    result.problem.generator = inp::OpticalOffloadGenerator{};

    // Optical limits
    if (auto steps = cfg.optical_limits().steps())
    {
        result.problem.limits.steps = steps;
    }
    if (auto step_iters = cfg.optical_limits().step_iters())
    {
        result.problem.limits.step_iters = step_iters;
    }

    // Optical capacities
    {
        auto const& ocfg = cfg.optical_capacity();
        result.problem.capacity.primaries = ocfg.primaries();
        result.problem.capacity.tracks = ocfg.tracks();
        result.problem.capacity.generators = ocfg.generators();
    }

    result.problem.num_streams = 1;
    result.problem.seed = cfg.seed();
    result.problem.timers.action = cfg.action_times();
    result.problem.output_file = cfg.output_file();

    return result;
}
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with fcl parameters.
 */
LarCelerStandalone::LarCelerStandalone(Parameters const& config)
    : runner_inp_{make_input_from_config(config())}
{
}

//---------------------------------------------------------------------------//
/*!
 * Start Celeritas at the beginning of the job.
 */
void LarCelerStandalone::beginJob()
{
    CELER_EXPECT(!runner_);
    runner_ = std::make_unique<LarStandaloneRunner>(
        std::forward<LarStandaloneRunner::Input>(runner_inp_));
}

//---------------------------------------------------------------------------//
/*!
 * Run Celeritas on a single event.
 */
auto LarCelerStandalone::executeEvent(VecSED const& edeps) -> UPVecBTR
{
    CELER_EXPECT(runner_);
    CELER_EXPECT(!edeps.empty());

    using VecBTR = LarStandaloneRunner::VecBTR;

    // Calculate detector responsors for the input steps
    auto& run = *runner_;
    VecBTR result = run(edeps);
    return std::make_unique<VecBTR>(std::move(result));
}

//---------------------------------------------------------------------------//
/*!
 * Free Celeritas memory at the end of the job.
 */
void LarCelerStandalone::endJob()
{
    CELER_EXPECT(runner_);
    runner_.reset();
}

//---------------------------------------------------------------------------//
/*!
 * No RNG initialization is needed.
 */
void LarCelerStandalone::InitializeTools(CLHEP::HepRandomEngine&,
                                         CLHEP::HepRandomEngine&)
{
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
