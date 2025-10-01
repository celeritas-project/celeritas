//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/SimParams.cc
//---------------------------------------------------------------------------//
#include "SimParams.hh"

#include <limits>

#include "corecel/Assert.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/field/FieldDriverOptions.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/phys/ParticleParams.hh"

#include "SimData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with imported data.
 */
SimParams::Input SimParams::Input::from_import(ImportData const& data,
                                               SPConstParticles particle_params)
{
    return SimParams::Input::from_import(
        data, std::move(particle_params), inp::TrackingLimits{});
}

//---------------------------------------------------------------------------//
/*!
 * Construct with imported data.
 */
SimParams::Input
SimParams::Input::from_import(ImportData const& data,
                              SPConstParticles particle_params,
                              inp::TrackingLimits const& limits)
{
    CELER_EXPECT(particle_params);
    CELER_EXPECT(data.trans_params);
    CELER_EXPECT(data.trans_params.looping.size() == particle_params->size());

    using MaxSubstepsInt = decltype(FieldDriverOptions{}.max_substeps);

    CELER_VALIDATE(limits.field_substeps > 0
                       && limits.field_substeps
                              <= std::numeric_limits<MaxSubstepsInt>::max(),
                   << "maximum field substep limit " << limits.field_substeps
                   << " is out of range (should be in (0, "
                   << std::numeric_limits<MaxSubstepsInt>::max() << "])");

    SimParams::Input input;
    input.particles = std::move(particle_params);

    // Calculate the maximum number of steps a track below the threshold energy
    // can take while looping (ceil(max Geant4 field propagator substeps / max
    // Celeritas field propagator substeps))
    CELER_ASSERT(data.trans_params.max_substeps
                 >= static_cast<int>(limits.field_substeps));
    auto max_subthreshold_steps = ceil_div<size_type>(
        data.trans_params.max_substeps, limits.field_substeps);

    for (auto pid : range(ParticleId{input.particles->size()}))
    {
        auto pdg = input.particles->id_to_pdg(pid);
        auto iter = data.trans_params.looping.find(pdg.get());
        CELER_ASSERT(iter != data.trans_params.looping.end());

        // Store the parameters for this particle
        LoopingThreshold looping;
        looping.max_subthreshold_steps = max_subthreshold_steps;
        looping.max_steps = iter->second.threshold_trials
                            * max_subthreshold_steps;
        looping.threshold_energy
            = LoopingThreshold::Energy(iter->second.important_energy);
        input.looping.insert({pdg, looping});
    }
    input.max_steps = limits.steps;

    return input;
}

//---------------------------------------------------------------------------//
/*!
 * Construct with simulation options.
 */
SimParams::SimParams(Input const& input)
{
    CELER_EXPECT(input.particles);
    CELER_VALIDATE(
        input.max_steps > 0
            && input.max_steps <= std::numeric_limits<size_type>::max(),
        << "maximum step limit " << input.max_steps
        << " is out of range (should be in (0, "
        << std::numeric_limits<size_type>::max() << "])");

    HostVal<SimParamsData> host_data;

    // Initialize with the default threshold values
    std::vector<LoopingThreshold> looping(input.particles->size());
    for (auto pid : range(ParticleId{input.particles->size()}))
    {
        auto pdg = input.particles->id_to_pdg(pid);
        auto iter = input.looping.find(pdg);
        if (iter != input.looping.end())
        {
            looping[pid.get()] = iter->second;
        }
        CELER_ASSERT(looping.back());
    }
    make_builder(&host_data.looping).insert_back(looping.begin(), looping.end());
    host_data.max_steps = input.max_steps;

    data_ = CollectionMirror<SimParamsData>{std::move(host_data)};
    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
