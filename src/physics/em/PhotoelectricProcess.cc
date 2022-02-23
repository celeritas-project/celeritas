//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PhotoelectricProcess.cc
//---------------------------------------------------------------------------//
#include "PhotoelectricProcess.hh"

#include <utility>

#include "base/Assert.hh"
#include "io/ImportLivermorePE.hh"
#include "io/ImportProcess.hh"
#include "io/LivermorePEReader.hh"
#include "physics/base/PDGNumber.hh"

#include "LivermorePEModel.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from host data.
 */
PhotoelectricProcess::PhotoelectricProcess(SPConstParticles particles,
                                           SPConstMaterials materials,
                                           SPConstImported  process_data)
    : particles_(std::move(particles))
    , materials_(std::move(materials))
    , imported_(process_data,
                particles_,
                ImportProcessClass::photoelectric,
                {pdg::gamma()})
{
    CELER_EXPECT(particles_);
    CELER_EXPECT(materials_);
}

//---------------------------------------------------------------------------//
/*!
 * Construct the models associated with this process.
 */
auto PhotoelectricProcess::build_models(ModelIdGenerator next_id) const
    -> VecModel
{
    LivermorePEModel::ReadData load_data = LivermorePEReader();
    return {std::make_shared<LivermorePEModel>(
        next_id(), *particles_, *materials_, load_data)};
}

//---------------------------------------------------------------------------//
/*!
 * Get the interaction cross sections for the given energy range.
 */
auto PhotoelectricProcess::step_limits(Applicability applic) const
    -> StepLimitBuilders
{
    return imported_.step_limits(std::move(applic));
}

//---------------------------------------------------------------------------//
/*!
 * Type of process.
 */
ProcessType PhotoelectricProcess::type() const
{
    return ProcessType::electromagnetic_discrete;
}

//---------------------------------------------------------------------------//
/*!
 * Name of the process.
 */
std::string PhotoelectricProcess::label() const
{
    return "Photoelectric effect";
}

//---------------------------------------------------------------------------//
} // namespace celeritas
