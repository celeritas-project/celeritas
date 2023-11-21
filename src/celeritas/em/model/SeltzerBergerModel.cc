//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/SeltzerBergerModel.cc
//---------------------------------------------------------------------------//
#include "SeltzerBergerModel.hh"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "celeritas_config.h"
#include "corecel/cont/Range.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/grid/TwodGridData.hh"
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
    : imported_(data,
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
    host_data.ids.action = id;
    host_data.ids.electron = particles.find(pdg::electron());
    host_data.ids.positron = particles.find(pdg::positron());
    host_data.ids.gamma = particles.find(pdg::gamma());
    CELER_VALIDATE(host_data.ids,
                   << "missing electron, positron, and/or gamma particles "
                      "(required for "
                   << this->description() << ")");

    // Save particle properties
    host_data.electron_mass = particles.get(host_data.ids.electron).mass();

    // Load differential cross sections
    make_builder(&host_data.differential_xs.elements)
        .reserve(materials.num_elements());
    for (auto el_id : range(ElementId{materials.num_elements()}))
    {
        auto element = materials.get(el_id);
        this->append_table(load_sb_table(element.atomic_number()),
                           &host_data.differential_xs);
    }
    CELER_ASSERT(host_data.differential_xs.elements.size()
                 == materials.num_elements());

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
    // TODO: potentially set lower energy bound based on (material-dependent)
    // BremsstrahlungProcess lambda table energy grid to avoid invoking the
    // interactor for tracks with energy below the interaction threshold

    Applicability electron_applic;
    electron_applic.particle = this->host_ref().ids.electron;
    electron_applic.lower = zero_quantity();
    electron_applic.upper = detail::seltzer_berger_limit();

    Applicability positron_applic = electron_applic;
    positron_applic.particle = this->host_ref().ids.positron;

    return {electron_applic, positron_applic};
}

//---------------------------------------------------------------------------//
/*!
 * Get the microscopic cross sections for the given particle and material.
 */
auto SeltzerBergerModel::micro_xs(Applicability applic) const -> MicroXsBuilders
{
    return imported_.micro_xs(std::move(applic));
}

//---------------------------------------------------------------------------//
/*!
 * Interact with host data.
 */
void SeltzerBergerModel::execute(CoreParams const& params,
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
void SeltzerBergerModel::execute(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
/*!
 * Get the model ID for this model.
 */
ActionId SeltzerBergerModel::action_id() const
{
    return this->host_ref().ids.action;
}

//---------------------------------------------------------------------------//
/*!
 * Construct differential cross section tables for a single element.
 *
 * Here, x = log of scaled incident energy (E / MeV)
 * and y = scaled exiting energy (E_gamma / E_inc)
 * and values are the cross sections.
 */
void SeltzerBergerModel::append_table(ImportSBTable const& imported,
                                      HostXsTables* tables) const
{
    auto reals = make_builder(&tables->reals);

    CELER_ASSERT(!imported.value.empty()
                 && imported.value.size()
                        == imported.x.size() * imported.y.size());
    const size_type num_x = imported.x.size();
    const size_type num_y = imported.y.size();

    SBElementTableData table;

    // TODO: hash the energy grid for reuse, because only Z = 100 has a
    // different energy grid.

    // Incident charged particle log energy grid
    table.grid.x = reals.insert_back(imported.x.begin(), imported.x.end());

    // Photon reduced energy grid
    table.grid.y = reals.insert_back(imported.y.begin(), imported.y.end());

    // 2D scaled DCS grid
    table.grid.values
        = reals.insert_back(imported.value.begin(), imported.value.end());

    // Find the location of the highest cross section at each incident E
    std::vector<size_type> argmax(table.grid.x.size());
    for (size_type i : range(num_x))
    {
        // Get the xs data for the given incident energy coordinate
        real_type const* iter = &tables->reals[table.grid.at(i, 0)];

        // Search for the highest cross section value
        size_type max_el = std::max_element(iter, iter + num_y) - iter;
        CELER_ASSERT(max_el < num_y);
        // Save it!
        argmax[i] = max_el;
    }
    table.argmax
        = make_builder(&tables->sizes).insert_back(argmax.begin(), argmax.end());

    // Add the table
    make_builder(&tables->elements).push_back(table);

    CELER_ENSURE(table.grid.x.size() == num_x);
    CELER_ENSURE(table.grid.y.size() == num_y);
    CELER_ENSURE(table.argmax.size() == num_x);
    CELER_ENSURE(table.grid);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
