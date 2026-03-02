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
#include "celeritas/inp/StandaloneInput.hh"
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
 * The optical material is determined in Celeritas when the tracks are
 * initialized from the pre-step position.
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
}  // namespace celeritas
