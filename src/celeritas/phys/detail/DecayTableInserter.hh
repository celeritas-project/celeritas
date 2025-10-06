//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/detail/DecayTableInserter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <unordered_map>
#include <vector>

#include "corecel/data/CollectionBuilder.hh"
#include "celeritas/Types.hh"
#include "celeritas/decay/DecayData.hh"
#include "celeritas/decay/DecayProcess.hh"
#include "celeritas/decay/channel/DecayChannel.hh"
#include "celeritas/inp/Physics.hh"
#include "celeritas/phys/PhysicsData.hh"

namespace celeritas
{
class ParticleParams;

namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct a decay table from input data.
 */
class DecayTableInserter
{
  public:
    //!@{
    //! \name Type aliases
    using Data = HostVal<PhysicsParamsData>;
    using DecayTable = inp::DecayPhysics::DecayTable;
    using SPConstParticles = std::shared_ptr<ParticleParams const>;
    using VecChannel = DecayProcess::VecChannel;
    //!@}

  public:
    // Construct with particles, decay channels and pointer to host data
    DecayTableInserter(SPConstParticles, VecChannel const&, Data&);

    // Construct decay table for a single particle
    DecayTableData operator()(DecayTable const& inp);

  private:
    using DCT = DecayChannelType;
    using MapChannelAction = std::unordered_map<DecayChannelType, ActionId>;

    SPConstParticles particles_;
    CollectionBuilder<real_type> reals_;
    CollectionBuilder<ParticleId> daughters_;
    CollectionBuilder<DecayChannelId> channel_ids_;
    CollectionBuilder<DecayChannelData, MemSpace::host, DecayChannelId> channels_;
    CollectionBuilder<ActionId, MemSpace::host, DecayChannelId> actions_;
    MapChannelAction channel_to_action_;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
