//------------------------------ -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/detail/DecayTableInserter.cc
//---------------------------------------------------------------------------//
#include "DecayTableInserter.hh"

#include "corecel/io/Logger.hh"
#include "celeritas/decay/channel/MuDecayChannel.hh"
#include "celeritas/phys/ParticleParams.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with particles, decay channels and pointer to host data.
 */
DecayTableInserter::DecayTableInserter(SPConstParticles particles,
                                       VecChannel const& channels,
                                       Data& data)
    : particles_(particles)
    , reals_{&data.reals}
    , daughters_{&data.daughters}
    , channel_ids_{&data.channel_ids}
    , channels_{&data.channels}
    , actions_{&data.actions}
{
    CELER_EXPECT(particles_);

    // Build a mapping of channel type to action ID
    for (auto const& channel : channels)
    {
        if (dynamic_cast<MuDecayChannel const*>(channel.get()))
        {
            channel_to_action_.insert({DCT::muon, channel->action_id()});
        }
        else
        {
            CELER_NOT_IMPLEMENTED("Decay channels other than muon");
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Construct decay table for a single particle.
 */
DecayTableData DecayTableInserter::operator()(DecayTable const& inp)
{
    if (inp.empty())
    {
        // No decay process for this particle
        return {};
    }

    DecayTableData result;

    double accum_br{0};
    std::vector<real_type> branching_ratios;
    std::vector<DecayChannelId> channel_ids;
    std::vector<DecayChannelData> channels;
    std::vector<ActionId> actions;

    branching_ratios.reserve(inp.size());
    channel_ids.reserve(inp.size());
    channels.reserve(inp.size());
    actions.reserve(inp.size());

    for (auto ch_inp : inp)
    {
        CELER_VALIDATE(ch_inp.type != DCT::size_,
                       << "invalid decay channel type");
        CELER_VALIDATE(ch_inp.branching_ratio > 0,
                       << "invalid branching_ratio=" << ch_inp.branching_ratio
                       << " (should be positive)");
        CELER_VALIDATE(!ch_inp.daughters.empty(),
                       << "decay channel must have daughters");

        // Get the particle ID from the PDG and store the daughters
        DecayChannelData channel;
        std::vector<ParticleId> daughters;
        daughters.reserve(ch_inp.daughters.size());
        for (auto pdg : ch_inp.daughters)
        {
            daughters.push_back(particles_->find(pdg));
        }
        channel.daughters
            = daughters_.insert_back(daughters.begin(), daughters.end());
        channel_ids.push_back(channels_.push_back(channel));

        // Store the branching ratio and find the channel's action ID
        branching_ratios.push_back(ch_inp.branching_ratio);
        actions.push_back(channel_to_action_[ch_inp.type]);

        // Calculate the sum of the branching ratios
        accum_br += ch_inp.branching_ratio;
    }
    result.channel_ids
        = channel_ids_.insert_back(channel_ids.begin(), channel_ids.end());

    if (!(soft_equal<double>(1, accum_br)))
    {
        CELER_LOG(warning)
            << "branching ratios for decay channels should sum to 1 "
               "but instead sum tp "
            << accum_br;
    }
    double norm = 1 / accum_br;
    for (auto& br : branching_ratios)
    {
        // Renormalize the branching ratios
        br *= norm;
    }
    result.branching_ratios
        = reals_.insert_back(branching_ratios.begin(), branching_ratios.end());

    // Append to mapping from channel ID to action ID
    actions_.insert_back(actions.begin(), actions.end());

    CELER_ENSURE(channels_.size() == actions_.size());
    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
