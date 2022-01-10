//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file EIonizationProcess.cc
//---------------------------------------------------------------------------//
#include "EIonizationProcess.hh"

#include "MollerBhabhaModel.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct process from host data.
 */
EIonizationProcess::EIonizationProcess(SPConstParticles particles,
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
auto EIonizationProcess::build_models(ModelIdGenerator next_id) const
    -> VecModel
{
    return {std::make_shared<MollerBhabhaModel>(next_id(), *particles_)};
}

//---------------------------------------------------------------------------//
/*!
 * Get cross section values.
 */
auto EIonizationProcess::step_limits(Applicability applicability) const
    -> StepLimitBuilders
{
    return imported_.step_limits(std::move(applicability));
}

//---------------------------------------------------------------------------//
/*!
 * Type of process.
 */
ProcessType EIonizationProcess::type() const
{
    return ProcessType::electromagnetic_dedx;
}

//---------------------------------------------------------------------------//
/*!
 * Name of the process.
 */
std::string EIonizationProcess::label() const
{
    return "Electron/positron ionization";
}

//---------------------------------------------------------------------------//
} // namespace celeritas
