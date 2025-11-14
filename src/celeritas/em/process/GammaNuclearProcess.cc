//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/process/GammaNuclearProcess.cc
//---------------------------------------------------------------------------//
#include "GammaNuclearProcess.hh"

#include "corecel/Assert.hh"
#include "celeritas/em/model/GammaNuclearModel.hh"
#include "celeritas/phys/PDGNumber.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from host data.
 */
GammaNuclearProcess::GammaNuclearProcess(SPConstParticles particles,
                                         SPConstMaterials materials,
                                         ReadData load_data)
    : particles_(std::move(particles))
    , materials_(std::move(materials))
    , load_data_(std::move(load_data))
{
    CELER_EXPECT(particles_);
    CELER_EXPECT(materials_);
    CELER_EXPECT(load_data_);
}

//---------------------------------------------------------------------------//
/*!
 * Construct the models associated with this process.
 */
auto GammaNuclearProcess::build_models(ActionIdIter id) const -> VecModel
{
    return {std::make_shared<GammaNuclearModel>(
        *id++, *particles_, *materials_, load_data_)};
}

//---------------------------------------------------------------------------//
/*!
 * Get the interaction cross sections for the given energy range.
 */
auto GammaNuclearProcess::macro_xs(Applicability) const -> XsGrid
{
    return {};
}

//---------------------------------------------------------------------------//
/*!
 * Get the energy loss for the given energy range.
 */
auto GammaNuclearProcess::energy_loss(Applicability) const -> EnergyLossGrid
{
    return {};
}

//---------------------------------------------------------------------------//
/*!
 * Name of the process.
 */
std::string_view GammaNuclearProcess::label() const
{
    return "Gamma nuclear";
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
