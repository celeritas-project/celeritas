//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file BremsstrahlungProcess.cc
//---------------------------------------------------------------------------//
#include "BremsstrahlungProcess.hh"

#include <utility>
#include "base/Assert.hh"
#include "io/SeltzerBergerReader.hh"
#include "physics/base/PDGNumber.hh"
#include "SeltzerBergerModel.hh"
#include "RelativisticBremModel.hh"
#include "CombinedBremModel.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from host data.
 */
BremsstrahlungProcess::BremsstrahlungProcess(SPConstParticles particles,
                                             SPConstMaterials materials,
                                             SPConstImported  process_data,
                                             Options          options)
    : particles_(std::move(particles))
    , materials_(std::move(materials))
    , imported_(process_data,
                particles_,
                ImportProcessClass::e_brems,
                {pdg::electron(), pdg::positron()})
    , options_(options)
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
    if (options_.combined_model)
    {
        return {std::make_shared<CombinedBremModel>(
            next_id(), *particles_, *materials_, load_data, options_.enable_lpm)};
    }
    else
    {
        return {std::make_shared<SeltzerBergerModel>(
                    next_id(), *particles_, *materials_, load_data),
                std::make_shared<RelativisticBremModel>(
                    next_id(), *particles_, *materials_, options_.enable_lpm)};
    }
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
    return ProcessType::electromagnetic_dedx;
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
