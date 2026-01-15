//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mucf/process/MucfProcess.cc
//---------------------------------------------------------------------------//
#include "MucfProcess.hh"

#include <memory>

#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/mucf/model/DTMixMucfModel.hh"
#include "celeritas/phys/Model.hh"
#include "celeritas/phys/ParticleParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from host data.
 */
MucfProcess::MucfProcess(SPConstParticles particles, SPConstMaterials materials)
    : particles_(particles), materials_(materials)
{
    //! \todo Fix ImportProcessClass
    CELER_EXPECT(particles_);
    CELER_EXPECT(materials_);
}

//---------------------------------------------------------------------------//
/*!
 * Construct the models associated with this process.
 */
auto MucfProcess::build_models(ActionIdIter start_id) const -> VecModel
{
    return {std::make_shared<DTMixMucfModel>(
        *start_id++, *particles_, *materials_)};
}

//---------------------------------------------------------------------------//
/*!
 * Get the interaction cross sections for the given energy range.
 */
auto MucfProcess::macro_xs(Applicability) const -> XsGrid
{
    return {};
}

//---------------------------------------------------------------------------//
/*!
 * Get the energy loss for the given energy range.
 */
auto MucfProcess::energy_loss(Applicability) const -> EnergyLossGrid
{
    return {};
}
//---------------------------------------------------------------------------//
/*!
 * Name of the process.
 */
std::string_view MucfProcess::label() const
{
    return "Muon-catalyzed fusion";
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
