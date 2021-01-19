//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file EPlusAnnihilationProcess.cc
//---------------------------------------------------------------------------//
#include "EPlusAnnihilationProcess.hh"

#include <utility>
#include "EPlusGGModel.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from host data.
 */
EPlusAnnihilationProcess::EPlusAnnihilationProcess(SPConstParticles particles)
    : particles_(std::move(particles))
    , positron_id_(particles_->find(pdg::positron()))
{
    CELER_EXPECT(particles_);
}

//---------------------------------------------------------------------------//
/*!
 * Construct the models associated with this process.
 */
auto EPlusAnnihilationProcess::build_models(ModelIdGenerator next_id) const
    -> VecModel
{
    return {std::make_shared<EPlusGGModel>(next_id(), *particles_)};
}

//---------------------------------------------------------------------------//
/*!
 * Get the interaction cross sections for the given energy range.
 */
auto EPlusAnnihilationProcess::step_limits(Applicability range) const
    -> StepLimitBuilders
{
    CELER_EXPECT(range.particle == positron_id_);

    // Not implemented
    CELER_ASSERT_UNREACHABLE();

    return {};
}

//---------------------------------------------------------------------------//
/*!
 * Name of the process.
 */
std::string EPlusAnnihilationProcess::label() const
{
    return "Positron annihiliation";
}

//---------------------------------------------------------------------------//
} // namespace celeritas
