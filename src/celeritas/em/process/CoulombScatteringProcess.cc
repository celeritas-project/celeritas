//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/process/CoulombScatteringProcess.cc
//---------------------------------------------------------------------------//
#include "CoulombScatteringProcess.hh"

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "celeritas/em/model/CoulombScatteringModel.hh"
#include "celeritas/io/ImportProcess.hh"
#include "celeritas/phys/PDGNumber.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from host data.
 */
CoulombScatteringProcess::CoulombScatteringProcess(SPConstParticles particles,
                                                   SPConstMaterials materials,
                                                   SPConstImported process_data)
    : particles_(std::move(particles))
    , materials_(std::move(materials))
    , imported_(process_data,
                particles_,
                ImportProcessClass::coulomb_scat,
                {pdg::electron(), pdg::positron()})
{
    CELER_EXPECT(particles_);
    CELER_EXPECT(materials_);
}

//---------------------------------------------------------------------------//
/*!
 * Construct the models associated with this process.
 */
auto CoulombScatteringProcess::build_models(ActionIdIter start_id) const -> VecModel
{
    return {std::make_shared<CoulombScatteringModel>(
        *start_id++, *particles_, *materials_, imported_.processes())};
}

//---------------------------------------------------------------------------//
/*!
 * Get the interaction cross sections for the given energy range.
 */
auto CoulombScatteringProcess::step_limits(Applicability applic) const
    -> StepLimitBuilders
{
    return imported_.step_limits(std::move(applic));
}

//---------------------------------------------------------------------------//
/*!
 * Name of the process.
 */
std::string_view CoulombScatteringProcess::label() const
{
    return "Coulomb scattering";
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
