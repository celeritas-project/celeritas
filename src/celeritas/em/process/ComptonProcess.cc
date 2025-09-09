//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/process/ComptonProcess.cc
//---------------------------------------------------------------------------//
#include "ComptonProcess.hh"

#include <utility>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "celeritas/em/model/KleinNishinaModel.hh"
#include "celeritas/io/ImportProcess.hh"
#include "celeritas/phys/PDGNumber.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from particles and imported Geant data.
 */
ComptonProcess::ComptonProcess(SPConstParticles particles,
                               SPConstImported process_data)
    : particles_(std::move(particles))
    , imported_(
          process_data, particles_, ImportProcessClass::compton, {pdg::gamma()})
{
    CELER_EXPECT(particles_);
}

//---------------------------------------------------------------------------//
/*!
 * Construct the models associated with this process.
 */
auto ComptonProcess::build_models(ActionIdIter start_id) const -> VecModel
{
    return {std::make_shared<KleinNishinaModel>(*start_id++, *particles_)};
}

//---------------------------------------------------------------------------//
/*!
 * Get the interaction cross sections for the given energy range.
 */
auto ComptonProcess::macro_xs(Applicability applic) const -> XsGrid
{
    return imported_.macro_xs(std::move(applic));
}

//---------------------------------------------------------------------------//
/*!
 * Get the energy loss for the given energy range.
 */
auto ComptonProcess::energy_loss(Applicability applic) const -> EnergyLossGrid
{
    return imported_.energy_loss(std::move(applic));
}

//---------------------------------------------------------------------------//
/*!
 * Name of the process.
 */
std::string_view ComptonProcess::label() const
{
    return "Compton scattering";
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
