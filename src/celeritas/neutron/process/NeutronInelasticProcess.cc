//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/neutron/process/NeutronInelasticProcess.cc
//---------------------------------------------------------------------------//
#include "NeutronInelasticProcess.hh"

#include "corecel/Assert.hh"
#include "celeritas/neutron/model/CascadeOptions.hh"
#include "celeritas/neutron/model/NeutronInelasticModel.hh"
#include "celeritas/phys/PDGNumber.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from host data.
 */
NeutronInelasticProcess::NeutronInelasticProcess(SPConstParticles particles,
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
auto NeutronInelasticProcess::build_models(ActionIdIter id) const -> VecModel
{
    CascadeOptions options;  // TODO: options as an argument of this process
    return {std::make_shared<NeutronInelasticModel>(
        *id++, *particles_, *materials_, options, load_data_)};
}

//---------------------------------------------------------------------------//
/*!
 * Get the interaction cross sections for the given energy range.
 */
auto NeutronInelasticProcess::macro_xs(Applicability) const -> XsGrid
{
    return {};
}

//---------------------------------------------------------------------------//
/*!
 * Get the energy loss for the given energy range.
 */
auto NeutronInelasticProcess::energy_loss(Applicability) const -> EnergyLossGrid
{
    return {};
}

//---------------------------------------------------------------------------//
/*!
 * Name of the process.
 */
std::string_view NeutronInelasticProcess::label() const
{
    return "Neutron inelastic";
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
