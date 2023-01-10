//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/process/EPlusAnnihilationProcess.cc
//---------------------------------------------------------------------------//
#include "EPlusAnnihilationProcess.hh"

#include <memory>
#include <type_traits>
#include <utility>

#include "corecel/cont/Range.hh"
#include "celeritas/em/model/EPlusGGModel.hh"
#include "celeritas/grid/ValueGridBuilder.hh"
#include "celeritas/grid/ValueGridData.hh"
#include "celeritas/phys/Model.hh"
#include "celeritas/phys/PDGNumber.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from host data.
 */
EPlusAnnihilationProcess::EPlusAnnihilationProcess(SPConstParticles particles,
                                                   Options options)
    : particles_(std::move(particles))
    , positron_id_(particles_->find(pdg::positron()))
    , options_(options)
{
    CELER_EXPECT(particles_);
    CELER_ENSURE(positron_id_);
}

//---------------------------------------------------------------------------//
/*!
 * Construct the models associated with this process.
 */
auto EPlusAnnihilationProcess::build_models(ActionIdIter start_id) const
    -> VecModel
{
    return {std::make_shared<EPlusGGModel>(*start_id++, *particles_)};
}

//---------------------------------------------------------------------------//
/*!
 * Get the interaction cross sections for the given energy range.
 */
auto EPlusAnnihilationProcess::step_limits(Applicability range) const
    -> StepLimitBuilders
{
    CELER_EXPECT(range.particle == positron_id_);

    StepLimitBuilders builders;
    builders[ValueGridType::macro_xs] = std::make_unique<ValueGridOTFBuilder>();

    return builders;
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
}  // namespace celeritas
