//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/process/RayleighProcess.cc
//---------------------------------------------------------------------------//
#include "RayleighProcess.hh"

#include <utility>

#include "celeritas/em/model/RayleighModel.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from host data.
 */
RayleighProcess::RayleighProcess(SPConstParticles particles,
                                 SPConstMaterials materials,
                                 SPConstImported  process_data)
    : particles_(std::move(particles))
    , materials_(std::move(materials))
    , imported_(
          process_data, particles_, ImportProcessClass::rayleigh, {pdg::gamma()})
{
    CELER_EXPECT(particles_);
    CELER_EXPECT(materials_);
}

//---------------------------------------------------------------------------//
/*!
 * Construct the models associated with this process.
 */
auto RayleighProcess::build_models(ActionIdIter start_id) const -> VecModel
{
    return {std::make_shared<RayleighModel>(
        *start_id++, *particles_, *materials_, imported_.processes())};
}

//---------------------------------------------------------------------------//
/*!
 * Get the interaction cross sections for the given energy range.
 */
auto RayleighProcess::step_limits(Applicability applic) const
    -> StepLimitBuilders
{
    return imported_.step_limits(std::move(applic));
}

//---------------------------------------------------------------------------//
/*!
 * Type of process.
 */
ProcessType RayleighProcess::type() const
{
    return ProcessType::electromagnetic_discrete;
}

//---------------------------------------------------------------------------//
/*!
 * Name of the process.
 */
std::string RayleighProcess::label() const
{
    return "Rayleigh scattering";
}

//---------------------------------------------------------------------------//
} // namespace celeritas
