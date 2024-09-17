//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/process/MuIonizationProcess.cc
//---------------------------------------------------------------------------//
#include "MuIonizationProcess.hh"

#include <utility>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "celeritas/em/model/BraggModel.hh"
#include "celeritas/em/model/ICRU73QOModel.hh"
#include "celeritas/em/model/MuBetheBlochModel.hh"
#include "celeritas/io/ImportProcess.hh"
#include "celeritas/phys/PDGNumber.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct process from host data.
 */
MuIonizationProcess::MuIonizationProcess(SPConstParticles particles,
                                         SPConstImported process_data,
                                         Options options)
    : particles_(std::move(particles))
    , imported_(process_data,
                particles_,
                ImportProcessClass::mu_ioni,
                {pdg::mu_minus(), pdg::mu_plus()})
    , options_(options)
{
    CELER_EXPECT(particles_);
}

//---------------------------------------------------------------------------//
/*!
 * Construct the models associated with this process.
 */
auto MuIonizationProcess::build_models(ActionIdIter start_id) const -> VecModel
{
    return {std::make_shared<ICRU73QOModel>(*start_id++, *particles_),
            std::make_shared<BraggModel>(*start_id++, *particles_),
            std::make_shared<MuBetheBlochModel>(*start_id++, *particles_)};
}

//---------------------------------------------------------------------------//
/*!
 * Get cross section values.
 */
auto MuIonizationProcess::step_limits(Applicability applicability) const
    -> StepLimitBuilders
{
    return imported_.step_limits(std::move(applicability));
}

//---------------------------------------------------------------------------//
/*!
 * Name of the process.
 */
std::string_view MuIonizationProcess::label() const
{
    return "Muon ionization";
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
