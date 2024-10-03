//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/process/MuBremsstrahlungProcess.cc
//---------------------------------------------------------------------------//
#include "MuBremsstrahlungProcess.hh"

#include <utility>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "celeritas/em/model/MuBremsstrahlungModel.hh"
#include "celeritas/io/ImportProcess.hh"
#include "celeritas/phys/PDGNumber.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from host data.
 */
MuBremsstrahlungProcess::MuBremsstrahlungProcess(SPConstParticles particles,
                                                 SPConstImported process_data,
                                                 Options options)
    : particles_(std::move(particles))
    , imported_(process_data,
                particles_,
                ImportProcessClass::mu_brems,
                {pdg::mu_minus(), pdg::mu_plus()})
    , options_(options)
{
    CELER_EXPECT(particles_);
}

//---------------------------------------------------------------------------//
/*!
 * Construct the models associated with this process.
 */
auto MuBremsstrahlungProcess::build_models(ActionIdIter start_id) const -> VecModel
{
    return {std::make_shared<MuBremsstrahlungModel>(
        *start_id++, *particles_, imported_.processes())};
}

//---------------------------------------------------------------------------//
/*!
 * Get the interaction cross sections for the given energy range.
 */
auto MuBremsstrahlungProcess::step_limits(Applicability applic) const
    -> StepLimitBuilders
{
    return imported_.step_limits(std::move(applic));
}

//---------------------------------------------------------------------------//
/*!
 * Name of the process.
 */
std::string_view MuBremsstrahlungProcess::label() const
{
    return "Muon bremsstrahlung";
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
