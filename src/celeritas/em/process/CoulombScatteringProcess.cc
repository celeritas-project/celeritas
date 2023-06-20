//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/process/CoulombScatteringProcess.cc
//---------------------------------------------------------------------------//
#include "CoulombScatteringProcess.hh"

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "celeritas/em/model/WokviModel.hh"
#include "celeritas/io/ImportProcess.hh"
#include "celeritas/phys/PDGNumber.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
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

auto CoulombScatteringProcess::build_models(ActionIdIter start_id) const
    -> VecModel
{
    return {std::make_shared<WokviModel>(
        *start_id++, *particles_, *materials_, imported_.processes())};
}

auto CoulombScatteringProcess::step_limits(Applicability applic) const
    -> StepLimitBuilders
{
    return imported_.step_limits(std::move(applic));
}

std::string CoulombScatteringProcess::label() const
{
    return "Coulomb Scattering";
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
