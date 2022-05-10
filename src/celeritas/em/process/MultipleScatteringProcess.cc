//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/process/MultipleScatteringProcess.cc
//---------------------------------------------------------------------------//
#include "MultipleScatteringProcess.hh"

#include "celeritas/em/model/UrbanMscModel.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct process from host data.
 */
MultipleScatteringProcess::MultipleScatteringProcess(
    SPConstParticles particles,
    SPConstMaterials materials,
    SPConstImported  process_data)
    : particles_(std::move(particles))
    , materials_(std::move(materials))
    , imported_(process_data,
                particles_,
                ImportProcessClass::msc,
                {pdg::electron(), pdg::positron()})
{
    CELER_EXPECT(particles_);
}

//---------------------------------------------------------------------------//
/*!
 * Construct the models associated with this process.
 */
auto MultipleScatteringProcess::build_models(ActionIdIter start_id) const
    -> VecModel
{
    return {std::make_shared<UrbanMscModel>(
        *start_id++, *particles_, *materials_)};
}

//---------------------------------------------------------------------------//
/*!
 * Get cross section values.
 */
auto MultipleScatteringProcess::step_limits(Applicability applicability) const
    -> StepLimitBuilders
{
    return imported_.step_limits(std::move(applicability));
}

//---------------------------------------------------------------------------//
/*!
 * Type of process.
 */
ProcessType MultipleScatteringProcess::type() const
{
    return ProcessType::electromagnetic_msc;
}

//---------------------------------------------------------------------------//
/*!
 * Name of the process.
 */
std::string MultipleScatteringProcess::label() const
{
    return "Multiple scattering";
}

//---------------------------------------------------------------------------//
} // namespace celeritas
