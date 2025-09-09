//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/SeltzerBergerModel.cc
//---------------------------------------------------------------------------//
#include "SeltzerBergerModel.hh"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "corecel/Config.hh"

#include "corecel/cont/Range.hh"
#include "corecel/data/Collection.hh"
#include "corecel/sys/ScopedMem.hh"
#include "celeritas/em/data/ElectronBremsData.hh"
#include "celeritas/em/executor/SeltzerBergerExecutor.hh"  // IWYU pragma: associated
#include "celeritas/em/interactor/detail/PhysicsConstants.hh"
#include "celeritas/em/interactor/detail/SBPositronXsCorrector.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/TrackExecutor.hh"
#include "celeritas/io/ImportProcess.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/phys/InteractionApplier.hh"  // IWYU pragma: associated
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/ParticleView.hh"

#include "detail/SBTableInserter.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from model ID and other necessary data.
 */
SeltzerBergerModel::SeltzerBergerModel(ActionId id,
                                       ParticleParams const& particles,
                                       MaterialParams const& materials,
                                       SPConstImported data,
                                       ReadData load_sb_table)
    : StaticConcreteAction(
          id, "brems-sb", "interact by Seltzer-Berger bremsstrahlung")
    , imported_(data,
                particles,
                ImportProcessClass::e_brems,
                ImportModelClass::e_brems_sb,
                {pdg::electron(), pdg::positron()})
{
    CELER_EXPECT(id);
    CELER_EXPECT(load_sb_table);

    ScopedMem record_mem("SeltzerBergerModel.construct");

    HostVal<SeltzerBergerData> host_data;

    // Save IDs
    host_data.ids.electron = particles.find(pdg::electron());
    host_data.ids.positron = particles.find(pdg::positron());
    host_data.ids.gamma = particles.find(pdg::gamma());
    CELER_VALIDATE(host_data.ids,
                   << "missing particles (required for " << this->description()
                   << ")");

    // Save particle properties
    host_data.electron_mass = particles.get(host_data.ids.electron).mass();

    // Set the model high energy limit
    host_data.high_energy_limit
        = imported_.high_energy_limit(host_data.ids.electron);
    CELER_VALIDATE(host_data.high_energy_limit
                       == imported_.high_energy_limit(host_data.ids.positron),
                   << "Seltzer-Berger energy grid bounds are inconsistent "
                      "across particles");

    // Load differential cross sections
    detail::SBTableInserter insert_element(&host_data.differential_xs);
    for (auto el_id : range(ElementId{materials.num_elements()}))
    {
        AtomicNumber z = materials.get(el_id).atomic_number();
        insert_element(load_sb_table(z));
    }
    CELER_ASSERT(host_data.differential_xs.elements.size()
                 == materials.num_elements());

    for (auto el_id : range(ElementId(materials.num_elements())))
    {
        auto const& el = host_data.differential_xs.elements[el_id];
        auto max_energy
            = std::exp(host_data.differential_xs.reals[el.grid.x.back()]);
        CELER_VALIDATE(max_energy >= host_data.high_energy_limit.value(),
                       << "Seltzer-Berger high energy limit ("
                       << host_data.high_energy_limit.value()
                       << " MeV) is above the maximum incident energy ("
                       << max_energy
                       << " MeV) for the differential cross section grid");
    }

    // Move to mirrored data, copying to device
    data_ = CollectionMirror<SeltzerBergerData>{std::move(host_data)};

    CELER_ENSURE(this->data_);
}

//---------------------------------------------------------------------------//
/*!
 * Particle types and energy ranges that this model applies to.
 */
auto SeltzerBergerModel::applicability() const -> SetApplicability
{
    /*!
     * \todo Set lower energy bound based on (material-dependent)
     * BremsstrahlungProcess lambda table energy grid to avoid invoking the
     * interactor for tracks with energy below the interaction threshold.
     */

    Applicability electron;
    electron.particle = this->host_ref().ids.electron;
    electron.lower = zero_quantity();
    electron.upper = this->host_ref().high_energy_limit;

    Applicability positron = electron;
    positron.particle = this->host_ref().ids.positron;

    return {electron, positron};
}

//---------------------------------------------------------------------------//
/*!
 * Get the microscopic cross sections for the given particle and material.
 */
auto SeltzerBergerModel::micro_xs(Applicability applic) const -> XsTable
{
    return imported_.micro_xs(std::move(applic));
}

//---------------------------------------------------------------------------//
/*!
 * Interact with host data.
 */
void SeltzerBergerModel::step(CoreParams const& params,
                              CoreStateHost& state) const
{
    auto execute = make_action_track_executor(
        params.ptr<MemSpace::native>(),
        state.ptr(),
        this->action_id(),
        InteractionApplier{SeltzerBergerExecutor{this->host_ref()}});
    return launch_action(*this, params, state, execute);
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void SeltzerBergerModel::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
