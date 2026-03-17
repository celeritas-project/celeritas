//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/neutron/model/NeutronCaptureModel.cc
//---------------------------------------------------------------------------//
#include "NeutronCaptureModel.hh"

#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"
#include "celeritas/grid/NonuniformGridInserter.hh"
// #include "celeritas/io/ImportPhysicsTable.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/neutron/executor/NeutronCaptureExecutor.hh"
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
NeutronCaptureModel::NeutronCaptureModel(ActionId id,
                                         ParticleParams const& particles,
                                         MaterialParams const& materials,
                                         ReadData load_data)
    : StaticConcreteAction(id,
                           "neutron-radiative-capture",
                           "interact by neutron radiative capture")
{
    CELER_EXPECT(id);
    CELER_EXPECT(load_data);

    HostVal<NeutronCaptureData> data;

    // Save IDs
    data.neutron_id = particles.find(pdg::neutron());

    CELER_VALIDATE(data.neutron_id,
                   << "missing neutron particles (required for "
                   << this->description() << ")");

    // Load neutron capture cross section data
    NonuniformGridInserter insert_grid{&data.reals, &data.micro_xs};
    for (auto el_id : range(ElementId{materials.num_elements()}))
    {
        AtomicNumber z = materials.get(el_id).atomic_number();
        insert_grid(load_data(z));
    }
    CELER_ASSERT(data.micro_xs.size() == materials.num_elements());

    // Move to mirrored data, copying to device
    mirror_ = ParamsDataStore<NeutronCaptureData>{std::move(data)};
    CELER_ENSURE(this->mirror_);
}

//---------------------------------------------------------------------------//
/*!
 * Particle types and energy ranges that this model applies to.
 */
auto NeutronCaptureModel::applicability() const -> SetApplicability
{
    Applicability neutron_applic;
    neutron_applic.particle = this->host_ref().neutron_id;
    neutron_applic.lower = zero_quantity();
    neutron_applic.upper = this->host_ref().max_valid_energy();

    return {neutron_applic};
}

//---------------------------------------------------------------------------//
/*!
 * Get the microscopic cross sections for the given particle and material.
 */
auto NeutronCaptureModel::micro_xs(Applicability) const -> XsTable
{
    // Cross sections are calculated on the fly
    return {};
}

//---------------------------------------------------------------------------//
//!@{
/*!
 * Apply the interaction kernel.
 */
void NeutronCaptureModel::step(CoreParams const& params,
                               CoreStateHost& state) const
{
    auto execute = make_action_track_executor(
        params.ptr<MemSpace::native>(),
        state.ptr(),
        this->action_id(),
        InteractionApplier{NeutronCaptureExecutor{this->host_ref()}});
    return launch_action(*this, params, state, execute);
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void NeutronCaptureModel::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
