//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GammaAnnihilationProcess.cc
//---------------------------------------------------------------------------//
#include "GammaAnnihilationProcess.hh"

#include <utility>
#include "BetheHeitlerModel.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from host data.
 */
GammaAnnihilationProcess::GammaAnnihilationProcess(SPConstParticles particles)
    : particles_(std::move(particles))
    , positron_id_(particles_->find(pdg::positron()))
{
    CELER_EXPECT(particles_);
}

//---------------------------------------------------------------------------//
/*!
 * Construct the models associated with this process.
 */
auto GammaAnnihilationProcess::build_models(ModelIdGenerator next_id) const
    -> VecModel
{
    return {std::make_shared<BetheHeitlerModel>(next_id(), *particles_)};
}

//---------------------------------------------------------------------------//
/*!
 * Get the interaction cross sections for the given energy range.
 */
auto GammaAnnihilationProcess::step_limits(Applicability range) const
    -> StepLimitBuilders
{
    CELER_EXPECT(range.particle == particles_->find(pdg::gamma()));

    // TODO
    StepLimitBuilders builders;
    return builders;
}

//---------------------------------------------------------------------------//
/*!
 * Name of the process.
 */
std::string GammaAnnihilationProcess::label() const
{
    return "Photon annihiliation";
}

//---------------------------------------------------------------------------//
} // namespace celeritas
