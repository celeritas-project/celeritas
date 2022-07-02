//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/process/BremsstrahlungProcess.cc
//---------------------------------------------------------------------------//
#include "BremsstrahlungProcess.hh"

#include <utility>

#include "corecel/Assert.hh"
#include "celeritas/em/model/CombinedBremModel.hh"
#include "celeritas/em/model/RelativisticBremModel.hh"
#include "celeritas/em/model/SeltzerBergerModel.hh"
#include "celeritas/io/SeltzerBergerReader.hh"
#include "celeritas/phys/PDGNumber.hh"

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
auto BremsstrahlungProcess::build_models(ActionIdIter start_id) const
    -> VecModel
{
    SeltzerBergerModel::ReadData load_data = SeltzerBergerReader();
    if (options_.combined_model)
    {
        // TODO: import micro xs for combined model
        return {std::make_shared<CombinedBremModel>(*start_id++,
                                                    *particles_,
                                                    *materials_,
                                                    imported_.processes(),
                                                    load_data,
                                                    options_.enable_lpm)};
    }
    else
    {
        return {std::make_shared<SeltzerBergerModel>(*start_id++,
                                                     *particles_,
                                                     *materials_,
                                                     imported_.processes(),
                                                     load_data),
                std::make_shared<RelativisticBremModel>(*start_id++,
                                                        *particles_,
                                                        *materials_,
                                                        imported_.processes(),
                                                        options_.enable_lpm)};
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
 * Name of the process.
 */
std::string BremsstrahlungProcess::label() const
{
    return "Bremsstrahlung";
}

//---------------------------------------------------------------------------//
} // namespace celeritas
