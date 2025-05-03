//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/process/EIonizationProcess.cc
//---------------------------------------------------------------------------//
#include "EIonizationProcess.hh"

#include <utility>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "celeritas/em/model/MollerBhabhaModel.hh"
#include "celeritas/io/ImportProcess.hh"
#include "celeritas/phys/PDGNumber.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct process from host data.
 */
EIonizationProcess::EIonizationProcess(SPConstParticles particles,
                                       SPConstImported process_data)
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
auto EIonizationProcess::build_models(ActionIdIter start_id) const -> VecModel
{
    return {std::make_shared<MollerBhabhaModel>(*start_id++, *particles_)};
}

//---------------------------------------------------------------------------//
/*!
 * Get the interaction cross sections for the given energy range.
 */
auto EIonizationProcess::macro_xs(Applicability applic) const -> XsGrid
{
    return imported_.macro_xs(std::move(applic));
}

//---------------------------------------------------------------------------//
/*!
 * Get the energy loss for the given energy range.
 */
auto EIonizationProcess::energy_loss(Applicability applic) const
    -> EnergyLossGrid
{
    return imported_.energy_loss(std::move(applic));
}

//---------------------------------------------------------------------------//
/*!
 * Name of the process.
 */
std::string_view EIonizationProcess::label() const
{
    return "Electron/positron ionization";
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
