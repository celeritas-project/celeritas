//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/neutron/model/NeutronInelasticModel.cc
//---------------------------------------------------------------------------//
#include "NeutronInelasticModel.hh"

#include "corecel/math/Quantity.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"
#include "celeritas/io/ImportPhysicsTable.hh"
#include "celeritas/io/ImportPhysicsVector.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/neutron/executor/NeutronInelasticExecutor.hh"
#include "celeritas/phys/InteractionApplier.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/ParticleView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from model ID and other necessary data.
 */
NeutronInelasticModel::NeutronInelasticModel(ActionId id,
                                             ParticleParams const& particles,
                                             MaterialParams const& materials,
                                             ReadData load_data)
{
    CELER_EXPECT(id);
    CELER_EXPECT(load_data);

    HostVal<NeutronInelasticData> data;

    // Save IDs
    data.scalars.action_id = id;
    data.scalars.neutron_id = particles.find(pdg::neutron());

    CELER_VALIDATE(data.scalars.neutron_id,
                   << "missing neutron particles (required for "
                   << this->description() << ")");

    // Save particle properties
    data.scalars.neutron_mass = particles.get(data.scalars.neutron_id).mass();

    // Load neutron elastic cross section data
    make_builder(&data.micro_xs).reserve(materials.num_elements());
    for (auto el_id : range(ElementId{materials.num_elements()}))
    {
        AtomicNumber z = materials.get(el_id).atomic_number();
        this->append_xs(load_data(z), &data);
    }
    CELER_ASSERT(data.micro_xs.size() == materials.num_elements());

    // Move to mirrored data, copying to device
    mirror_ = CollectionMirror<NeutronInelasticData>{std::move(data)};
    CELER_ENSURE(this->mirror_);
}

//---------------------------------------------------------------------------//
/*!
 * Particle types and energy ranges that this model applies to.
 */
auto NeutronInelasticModel::applicability() const -> SetApplicability
{
    Applicability neutron_applic;
    neutron_applic.particle = this->host_ref().scalars.neutron_id;
    neutron_applic.lower = this->host_ref().scalars.min_valid_energy();
    neutron_applic.upper = this->host_ref().scalars.max_valid_energy();

    return {neutron_applic};
}

//---------------------------------------------------------------------------//
/*!
 * Get the microscopic cross sections for the given particle and material.
 */
auto NeutronInelasticModel::micro_xs(Applicability) const -> MicroXsBuilders
{
    // Cross sections are calculated on the fly
    return {};
}

//---------------------------------------------------------------------------//
//!@{
/*!
 * Apply the interaction kernel.
 */
void NeutronInelasticModel::execute(CoreParams const&, CoreStateHost&) const
{
    CELER_NOT_IMPLEMENTED("Neutron inelastic interaction");
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void NeutronInelasticModel::execute(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
/*!
 * Get the acition ID for this model.
 */
ActionId NeutronInelasticModel::action_id() const
{
    return this->host_ref().scalars.action_id;
}

//---------------------------------------------------------------------------//
/*!
 * Construct neutron inelastic cross section data for a single element.
 */
void NeutronInelasticModel::append_xs(ImportPhysicsVector const& inp,
                                      HostXsData* data) const
{
    auto reals = make_builder(&data->reals);
    GenericGridData micro_xs;

    // Add the tabulated interaction cross section from input
    micro_xs.grid = reals.insert_back(inp.x.begin(), inp.x.end());
    micro_xs.value = reals.insert_back(inp.y.begin(), inp.y.end());
    micro_xs.grid_interp = Interp::linear;
    micro_xs.value_interp = Interp::linear;

    // Add micro xs data
    CELER_ASSERT(micro_xs);
    make_builder(&data->micro_xs).push_back(micro_xs);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
