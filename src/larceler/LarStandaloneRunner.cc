//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file larceler/LarStandaloneRunner.cc
//---------------------------------------------------------------------------//
#include "LarStandaloneRunner.hh"

#include <utility>
#include <lardataobj/Simulation/OpDetBacktrackerRecord.h>
#include <lardataobj/Simulation/SimEnergyDeposit.h>

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"
#include "geocel/GeantGeoParams.hh"
#include "celeritas/optical/Runner.hh"

#include "Convert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with input parameters.
 */
LarStandaloneRunner::LarStandaloneRunner(Input&& i)
    : runner_(std::make_shared<optical::Runner>(std::move(i)))
{
}

//---------------------------------------------------------------------------//
/*!
 * Run scintillation optical photons from a single set of energy steps.
 *
 * \todo With Cherenkov enabled we would need to determine the incident
 * particle's charge and the pre- and post-step speed.
 */
auto LarStandaloneRunner::operator()(VecSED const& sed) -> VecBTR
{
    CELER_EXPECT(!sed.empty());

    std::vector<celeritas::optical::GeneratorDistributionData> gdd;
    gdd.reserve(sed.size());

    for (auto const& edep : sed)
    {
        // Convert LArSoft sim edeps to Celeritas generator distribution data
        celeritas::optical::GeneratorDistributionData data;
        data.type = GeneratorType::scintillation;
        data.num_photons = edep.NumPhotons();
        data.primary = id_cast<PrimaryId>(edep.TrackID());
        data.step_length = convert_from_larsoft<LarsoftLen>(edep.StepLength());
        //! XXX Given post-step point find optical material
        data.material = OptMatId{0};
        // Assume continuous energy loss along the step
        //! \todo For neutral particles, set this to 0 (LED at post-step point)
        data.continuous_edep_fraction = 1;
        data.points[StepPoint::pre].time
            = convert_from_larsoft<LarsoftTime>(edep.StartT());
        data.points[StepPoint::pre].pos
            = convert_from_larsoft<LarsoftLen>(edep.Start());
        data.points[StepPoint::post].time
            = convert_from_larsoft<LarsoftTime>(edep.EndT());
        data.points[StepPoint::post].pos
            = convert_from_larsoft<LarsoftLen>(edep.End());
        CELER_ASSERT(data);
        gdd.push_back(data);
    }

    auto result = (*runner_)(make_span(std::as_const(gdd)));

    CELER_LOG(error) << "LArSoft interface is incomplete: no hits are "
                        "simulated";

    CELER_ASSERT(result.counters.generators.size() == 1);
    auto const& gen = result.counters.generators.front();
    CELER_LOG(debug) << "Transported " << gen.num_generated
                     << " optical photons from " << gen.buffer_size
                     << " sim energy deposits a total of "
                     << result.counters.steps << " steps over "
                     << result.counters.step_iters << " step iterations";

    return {};
}

//---------------------------------------------------------------------------//
/*!
 * Convert from a FHiCL config input.
 */
inp::OpticalStandaloneInput
from_config(detail::LarCelerStandaloneConfig const& cfg)
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

    result.problem.model.geometry = cfg.geometry();
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

//---------------------------------------------------------------------------//
}  // namespace celeritas
