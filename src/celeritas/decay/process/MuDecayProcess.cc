//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/decay/process/MuDecayProcess.cc
//---------------------------------------------------------------------------//
#include "MuDecayProcess.hh"

#include "corecel/Assert.hh"
#include "celeritas/decay/model/MuDecayModel.hh"
#include "celeritas/phys/PDGNumber.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from shared data.
 */
MuDecayProcess::MuDecayProcess(SPConstParticles particles)
    : particles_(particles)
{
    CELER_EXPECT(particles_);
}

//---------------------------------------------------------------------------//
/*!
 * Construct the models associated with this process.
 */
auto MuDecayProcess::build_models(ActionIdIter start_id) const -> VecModel
{
    return {std::make_shared<MuDecayModel>(*start_id++, *particles_)};
}

//---------------------------------------------------------------------------//
/*!
 * Get the interaction cross sections for the given energy range.
 */
auto MuDecayProcess::step_limits(Applicability applic) const -> StepLimitBuilders
{
    // TODO
}

//---------------------------------------------------------------------------//
/*!
 * Name of the process.
 */
std::string_view MuDecayProcess::label() const
{
    return "Muon decay process";
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
