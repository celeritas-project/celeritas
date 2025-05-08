//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/neutron/process/NeutronElasticProcess.cc
//---------------------------------------------------------------------------//
#include "NeutronElasticProcess.hh"

#include "corecel/Assert.hh"
#include "celeritas/neutron/model/ChipsNeutronElasticModel.hh"
#include "celeritas/phys/PDGNumber.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from host data.
 */
NeutronElasticProcess::NeutronElasticProcess(SPConstParticles particles,
                                             SPConstMaterials materials,
                                             ReadData load_data)
    : particles_(std::move(particles))
    , materials_(std::move(materials))
    , load_data_(std::move(load_data))
    , neutron_id_(particles_->find(pdg::neutron()))
{
    CELER_EXPECT(particles_);
    CELER_EXPECT(materials_);
    CELER_EXPECT(load_data_);
    CELER_ENSURE(neutron_id_);
}

//---------------------------------------------------------------------------//
/*!
 * Construct the models associated with this process.
 */
auto NeutronElasticProcess::build_models(ActionIdIter id) const -> VecModel
{
    return {std::make_shared<ChipsNeutronElasticModel>(
        *id++, *particles_, *materials_, load_data_)};
}

//---------------------------------------------------------------------------//
/*!
 * Get the interaction cross sections for the given energy range.
 */
auto NeutronElasticProcess::macro_xs(Applicability) const -> XsGrid
{
    return {};
}

//---------------------------------------------------------------------------//
/*!
 * Get the energy loss for the given energy range.
 */
auto NeutronElasticProcess::energy_loss(Applicability) const -> EnergyLossGrid
{
    return {};
}

//---------------------------------------------------------------------------//
/*!
 * Name of the process.
 */
std::string_view NeutronElasticProcess::label() const
{
    return "Neutron elastic";
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
