//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/SimParams.cc
//---------------------------------------------------------------------------//
#include "SimParams.hh"

#include "corecel/Assert.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/track/SimData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with imported data.
 */
std::shared_ptr<SimParams>
SimParams::from_import(ImportData const& data, SPConstParticles particle_params)
{
    CELER_EXPECT(data);
    CELER_EXPECT(particle_params);
    CELER_EXPECT(data.trans_params.size() == particle_params->size());

    SimParams::Input input;
    input.particles = std::move(particle_params);

    for (auto pid : range(ParticleId{input.particles->size()}))
    {
        auto pdg = input.particles->id_to_pdg(pid);
        auto iter = data.trans_params.find(pdg.get());
        CELER_ASSERT(iter != data.trans_params.end());

        // Store the parameters for this particle
        LoopingThreshold looping;
        looping.max_steps = iter->second.threshold_trials;
        looping.threshold_energy
            = LoopingThreshold::Energy{iter->second.important_energy};
        input.looping.insert({pdg, looping});
    }
    return std::make_shared<SimParams>(input);
}

//---------------------------------------------------------------------------//
/*!
 * Construct with simulation options.
 */
SimParams::SimParams(Input const& input)
{
    CELER_EXPECT(input.particles);

    HostValue host_data;

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

    data_ = CollectionMirror<SimParamsData>{std::move(host_data)};
    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
