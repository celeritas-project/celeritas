//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file BremsstrahlungProcess.cc
//---------------------------------------------------------------------------//
#include "BremsstrahlungProcess.hh"

#include <utility>
#include "io/SeltzerBergerReader.hh"
#include "SeltzerBergerModel.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from host data.
 */
BremsstrahlungProcess::BremsstrahlungProcess(SPConstParticles particles,
                                             SPConstMaterials materials,
                                             SPConstImported  process_data)
    : particles_(std::move(particles))
    , materials_(std::move(materials))
    , imported_(process_data,
                particles_,
                ImportProcessClass::e_brems,
                {pdg::electron(), pdg::positron()})
{
    CELER_EXPECT(particles_);
    CELER_EXPECT(materials_);
}

//---------------------------------------------------------------------------//
/*!
 * Construct the models associated with this process.
 */
auto BremsstrahlungProcess::build_models(ModelIdGenerator next_id) const
    -> VecModel
{
    SeltzerBergerModel::ReadData load_data = SeltzerBergerReader();
    return {std::make_shared<SeltzerBergerModel>(
        next_id(), *particles_, *materials_, load_data)};
}

//---------------------------------------------------------------------------//
/*!
 * Get the interaction cross sections for the given energy range.
 */
auto BremsstrahlungProcess::step_limits(Applicability applic) const
    -> StepLimitBuilders
{
    return imported_.step_limits(std::move(applic));
}

//---------------------------------------------------------------------------//
/*!
 * Type of process.
 */
ProcessType BremsstrahlungProcess::type() const
{
    return ProcessType::energy_loss;
}

//---------------------------------------------------------------------------//
/*!
 * Name of the process.
 */
std::string BremsstrahlungProcess::label() const
{
    return "Bremsstrahlung";
}

//---------------------------------------------------------------------------//
} // namespace celeritas
