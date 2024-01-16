//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/process/PhotoelectricProcess.cc
//---------------------------------------------------------------------------//
#include "PhotoelectricProcess.hh"

#include <utility>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "celeritas/em/model/LivermorePEModel.hh"
#include "celeritas/io/ImportProcess.hh"
#include "celeritas/phys/PDGNumber.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from host data.
 */
PhotoelectricProcess::PhotoelectricProcess(SPConstParticles particles,
                                           SPConstMaterials materials,
                                           SPConstImported process_data,
                                           ReadData load_data)
    : particles_(std::move(particles))
    , materials_(std::move(materials))
    , imported_(process_data,
                particles_,
                ImportProcessClass::photoelectric,
                {pdg::gamma()})
    , load_pe_(std::move(load_data))
{
    CELER_EXPECT(particles_);
    CELER_EXPECT(materials_);
    CELER_EXPECT(load_pe_);
}

//---------------------------------------------------------------------------//
/*!
 * Construct the models associated with this process.
 */
auto PhotoelectricProcess::build_models(ActionIdIter start_id) const -> VecModel
{
    return {std::make_shared<LivermorePEModel>(
        *start_id++, *particles_, *materials_, load_pe_)};
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
 * Name of the process.
 */
std::string PhotoelectricProcess::label() const
{
    return "Photoelectric effect";
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
