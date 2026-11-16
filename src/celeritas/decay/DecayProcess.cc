//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/decay/DecayProcess.cc
//---------------------------------------------------------------------------//
#include "DecayProcess.hh"

#include <utility>

#include "corecel/Assert.hh"
#include "celeritas/phys/ParticleParams.hh"

#include "channel/MuDecayChannel.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from particles and input.
 */
DecayProcess::DecayProcess(SPConstParticles particles, inp::DecayPhysics input)
    : particles_(std::move(particles)), input_(std::move(input))
{
    CELER_EXPECT(particles_);
    CELER_VALIDATE(input_, << "no decay channels are enabled");
}

//---------------------------------------------------------------------------//
/*!
 * Construct the decay channels, assigning each an action ID.
 */
auto DecayProcess::build_channels(ActionIdIter start_id) const -> VecChannel
{
    VecChannel result;
    if (input_.muon)
    {
        result.push_back(std::make_shared<MuDecayChannel>(
            *start_id++, *particles_, input_.muon));
    }
    CELER_ENSURE(!result.empty());
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
