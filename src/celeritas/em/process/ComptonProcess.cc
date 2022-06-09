//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/process/ComptonProcess.cc
//---------------------------------------------------------------------------//
#include "ComptonProcess.hh"

#include <utility>

#include "celeritas/em/model/KleinNishinaModel.hh"

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
auto ComptonProcess::build_models(ActionIdIter start_id) const -> VecModel
{
    return {std::make_shared<KleinNishinaModel>(
        *start_id++, *particles_, std::move(imported_.processes()))};
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
    return ProcessType::electromagnetic_discrete;
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
