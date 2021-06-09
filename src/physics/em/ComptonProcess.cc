//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ComptonProcess.cc
//---------------------------------------------------------------------------//
#include "ComptonProcess.hh"

#include <utility>
#include "KleinNishinaModel.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from particles and imported Geant data.
 */
ComptonProcess::ComptonProcess(SPConstParticles particles,
                               SPConstImported  process_data)
    : particles_(std::move(particles))
    , imported_(
          process_data, particles_, ImportProcessClass::compton, {pdg::gamma()})
{
    CELER_EXPECT(particles_);
}

//---------------------------------------------------------------------------//
/*!
 * Construct the models associated with this process.
 */
auto ComptonProcess::build_models(ModelIdGenerator next_id) const -> VecModel
{
    return {std::make_shared<KleinNishinaModel>(next_id(), *particles_)};
}

//---------------------------------------------------------------------------//
/*!
 * Get the interaction cross sections for the given energy range.
 */
auto ComptonProcess::step_limits(Applicability applic) const
    -> StepLimitBuilders
{
    return imported_.step_limits(std::move(applic));
}

//---------------------------------------------------------------------------//
/*!
 * Type of process.
 */
ProcessType ComptonProcess::type() const
{
    return ProcessType::electromagnetic;
}

//---------------------------------------------------------------------------//
/*!
 * Name of the process.
 */
std::string ComptonProcess::label() const
{
    return "Compton scattering";
}

//---------------------------------------------------------------------------//
} // namespace celeritas
