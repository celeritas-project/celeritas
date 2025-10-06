//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/decay/DecayProcess.cc
//---------------------------------------------------------------------------//
#include "DecayProcess.hh"

#include <set>

#include "corecel/Assert.hh"
#include "celeritas/decay/DecayData.hh"
#include "celeritas/decay/channel/MuDecayChannel.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from particles and input data.
 */
DecayProcess::DecayProcess(SPConstParticles particles,
                           inp::DecayPhysics const& input)
    : particles_(std::move(particles)), input_(input)
{
    CELER_EXPECT(particles_);
    CELER_VALIDATE(input_, << "no decay tables are present");
}

//---------------------------------------------------------------------------//
/*!
 * Construct the decay channels.
 */
auto DecayProcess::build_channels(ActionIdIter start_id) const -> VecChannel
{
    VecChannel result;
    std::set<DecayChannelType> types;
    for (auto const& [pdg, table] : input_.tables)
    {
        for (auto const& channel : table)
        {
            // Identify the unique channel types
            auto [iter, inserted] = types.insert(channel.type);
            if (inserted)
            {
                // Construct an action for each channel
                switch (*iter)
                {
                    case DecayChannelType::muon:
                        result.push_back(std::make_shared<MuDecayChannel>(
                            *start_id++, input_));
                        break;
                    default:
                        CELER_NOT_IMPLEMENTED("Decay channel type");
                }
            }
        }
    }
    CELER_VALIDATE(!result.empty(),
                   << "no supported channels for decay process");
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get the decay table for a particle.
 */
auto DecayProcess::decay_table(ParticleId pid) const -> DecayTable
{
    auto iter = input_.tables.find(particles_->id_to_pdg(pid));
    if (iter == input_.tables.end())
    {
        return {};
    }
    return iter->second;
}

//---------------------------------------------------------------------------//
/*!
 * Whether the decay process applies to the particle.
 */
bool DecayProcess::is_applicable(ParticleId pid) const
{
    return input_.tables.find(particles_->id_to_pdg(pid))
           != input_.tables.end();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
