//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
                                                 SPConstImported process_data)
    : particles_(std::move(particles))
    , imported_(process_data,
                particles_,
                ImportProcessClass::mu_brems,
                {pdg::mu_minus(), pdg::mu_plus()})
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
auto MuBremsstrahlungProcess::macro_xs(Applicability applic) const -> XsGrid
{
    return imported_.macro_xs(std::move(applic));
}

//---------------------------------------------------------------------------//
/*!
 * Get the energy loss for the given energy range.
 */
auto MuBremsstrahlungProcess::energy_loss(Applicability applic) const
    -> EnergyLossGrid
{
    return imported_.energy_loss(std::move(applic));
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
