//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/decay/channel/MuDecayChannel.cc
//---------------------------------------------------------------------------//
#include "MuDecayChannel.hh"

#include "celeritas/Quantities.hh"
#include "celeritas/decay/executor/MuDecayExecutor.hh"
#include "celeritas/em/interactor/detail/PhysicsConstants.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"
#include "celeritas/phys/InteractionApplier.hh"
#include "celeritas/phys/PDGNumber.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from action ID and input decay table data.
 */
MuDecayChannel::MuDecayChannel(ActionId id, inp::DecayPhysics const& input)
    : StaticConcreteAction(id, "muon-decay-channel", "muon decay")
{
    CELER_EXPECT(id);
    CELER_EXPECT(input);

    // Validate the input data
    for (auto const& [pdg, table] : input.tables)
    {
        for (auto const& channel : table)
        {
            // Loop over decay tables and find all channels of this type
            if (channel.type == DecayChannelType::muon)
            {
                // Check that the input data matches the data expected by the
                // muon decay interactor
                auto const& daughters = channel.daughters;
                CELER_VALIDATE(daughters.size() == 1,
                               << "muon decay only supports one daughter: "
                                  "neutrinos are currently neglected");
                CELER_VALIDATE(
                    (pdg == pdg::mu_minus() && daughters[0] == pdg::electron())
                        || (pdg == pdg::mu_plus()
                            && daughters[0] == pdg::positron()),
                    << "expected decay channel mu- -> e- or mu+ -> e+");
            }
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Interact with host data.
 */
void MuDecayChannel::step(CoreParams const& params, CoreStateHost& state) const
{
    auto execute
        = make_action_track_executor(params.ptr<MemSpace::native>(),
                                     state.ptr(),
                                     this->action_id(),
                                     InteractionApplier{MuDecayExecutor{}});
    return launch_action(*this, params, state, execute);
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void MuDecayChannel::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
