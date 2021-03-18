//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file EIonisationProcess.cc
//---------------------------------------------------------------------------//
#include "EIonisationProcess.hh"

#include "MollerBhabhaModel.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct process from host data.
 */
EIonisationProcess::EIonisationProcess(SPConstParticles particles,
                                       SPConstImported  process_data)
    : particles_(std::move(particles))
    , imported_(process_data,
                particles_,
                ImportProcessClass::e_ioni,
                {pdg::electron(), pdg::positron()})
{
    CELER_EXPECT(particles_);
}

//---------------------------------------------------------------------------//
/*!
 * Construct the models associated with this process.
 */
auto EIonisationProcess::build_models(ModelIdGenerator next_id) const
    -> VecModel
{
    return {std::make_shared<MollerBhabhaModel>(next_id(), *particles_)};
}

//---------------------------------------------------------------------------//
/*!
 * Get cross section values.
 */
auto EIonisationProcess::step_limits(Applicability applicability) const
    -> StepLimitBuilders
{
    return imported_.step_limits(std::move(applicability));
}

//---------------------------------------------------------------------------//
/*!
 * Name of the process.
 */
std::string EIonisationProcess::label() const
{
    return "Electron/positron ionisation";
}

//---------------------------------------------------------------------------//
} // namespace celeritas
