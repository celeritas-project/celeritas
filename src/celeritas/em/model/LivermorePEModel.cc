//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/LivermorePEModel.cc
//---------------------------------------------------------------------------//
#include "LivermorePEModel.hh"

#include <algorithm>
#include <utility>
#include <vector>

#include "corecel/Config.hh"

#include "corecel/cont/Range.hh"
#include "celeritas/em/data/LivermorePEData.hh"
#include "celeritas/em/executor/LivermorePEExecutor.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"
#include "celeritas/io/ImportLivermorePE.hh"
#include "celeritas/mat/ElementView.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/phys/InteractionApplier.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/ParticleView.hh"

#include "detail/LivermoreXsInserter.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from model ID and other necessary data.
 */
LivermorePEModel::LivermorePEModel(ActionId id,
                                   ParticleParams const& particles,
                                   MaterialParams const& materials,
                                   ReadData load_data)
    : StaticConcreteAction(
          id, "photoel-livermore", "interact by Livermore photoelectric effect")
{
    CELER_EXPECT(id);
    CELER_EXPECT(load_data);

    HostVal<LivermorePEData> host_data;

    // Save IDs
    host_data.ids.electron = particles.find(pdg::electron());
    host_data.ids.gamma = particles.find(pdg::gamma());
    CELER_VALIDATE(host_data.ids,
                   << R"(missing electron and/or gamma particles (required for )"
                   << this->description() << ")");

    // Save particle properties
    host_data.inv_electron_mass
        = 1
          / value_as<LivermorePERef::Mass>(
              particles.get(host_data.ids.electron).mass());

    // Load Livermore cross section data
    detail::LivermoreXsInserter insert_element(&host_data.xs);
    for (auto el_id : range(ElementId{materials.num_elements()}))
    {
        AtomicNumber z = materials.get(el_id).atomic_number();
        insert_element(load_data(z));
    }
    CELER_ASSERT(host_data.xs.elements.size() == materials.num_elements());

    // Move to mirrored data, copying to device
    data_ = CollectionMirror<LivermorePEData>{std::move(host_data)};
    CELER_ENSURE(this->data_);
}

//---------------------------------------------------------------------------//
/*!
 * Particle types and energy ranges that this model applies to.
 */
auto LivermorePEModel::applicability() const -> SetApplicability
{
    Applicability photon_applic;
    photon_applic.particle = this->host_ref().ids.gamma;
    photon_applic.lower = zero_quantity();
    photon_applic.upper = max_quantity();

    return {photon_applic};
}

//---------------------------------------------------------------------------//
/*!
 * Get the microscopic cross sections for the given particle and material.
 */
auto LivermorePEModel::micro_xs(Applicability) const -> XsTable
{
    // Cross sections are calculated on the fly
    return {};
}

//---------------------------------------------------------------------------//
//!@{
/*!
 * Interact with host data.
 */
void LivermorePEModel::step(CoreParams const& params, CoreStateHost& state) const
{
    auto execute = make_action_track_executor(
        params.ptr<MemSpace::native>(),
        state.ptr(),
        this->action_id(),
        InteractionApplier{LivermorePEExecutor{this->host_ref()}});
    return launch_action(*this, params, state, execute);
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void LivermorePEModel::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif
//!@}

//---------------------------------------------------------------------------//
}  // namespace celeritas
