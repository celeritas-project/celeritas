//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/process/ElectroNuclearProcess.cc
//---------------------------------------------------------------------------//
#include "ElectroNuclearProcess.hh"

#include "corecel/Assert.hh"
#include "celeritas/em/model/ElectroNuclearModel.hh"
#include "celeritas/phys/PDGNumber.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from host data.
 */
ElectroNuclearProcess::ElectroNuclearProcess(SPConstParticles particles,
                                             SPConstMaterials materials)
    : particles_(std::move(particles)), materials_(std::move(materials))
{
    CELER_EXPECT(particles_);
    CELER_EXPECT(materials_);
}

//---------------------------------------------------------------------------//
/*!
 * Construct the models associated with this process.
 */
auto ElectroNuclearProcess::build_models(ActionIdIter id) const -> VecModel
{
    return {std::make_shared<ElectroNuclearModel>(
        *id++, *particles_, *materials_)};
}

//---------------------------------------------------------------------------//
/*!
 * Get the interaction cross sections for the given energy range.
 */
auto ElectroNuclearProcess::macro_xs(Applicability) const -> XsGrid
{
    return {};
}

//---------------------------------------------------------------------------//
/*!
 * Get the energy loss for the given energy range.
 */
auto ElectroNuclearProcess::energy_loss(Applicability) const -> EnergyLossGrid
{
    return {};
}

//---------------------------------------------------------------------------//
/*!
 * Name of the process.
 */
std::string_view ElectroNuclearProcess::label() const
{
    return "Electro nuclear";
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
