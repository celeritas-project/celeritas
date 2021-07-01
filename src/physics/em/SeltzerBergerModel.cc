//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SeltzerBergerModel.cc
//---------------------------------------------------------------------------//
#include "SeltzerBergerModel.hh"

#include <algorithm>
#include "base/Assert.hh"
#include "base/CollectionBuilder.hh"
#include "comm/Logger.hh"
#include "base/Join.hh"
#include "base/Range.hh"
#include "physics/base/ParticleParams.hh"
#include "physics/base/PDGNumber.hh"
#include "physics/material/MaterialParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from model ID and other necessary data.
 */
SeltzerBergerModel::SeltzerBergerModel(ModelId               id,
                                       const ParticleParams& particles,
                                       const MaterialParams& materials,
                                       ReadData              load_sb_table)
{
    CELER_EXPECT(id);
    CELER_EXPECT(load_sb_table);

    detail::SeltzerBergerData<Ownership::value, MemSpace::host> host_data;

    // Save IDs
    host_data.ids.model    = id;
    host_data.ids.electron = particles.find(pdg::electron());
    host_data.ids.positron = particles.find(pdg::positron());
    host_data.ids.gamma    = particles.find(pdg::gamma());
    CELER_VALIDATE(host_data.ids,
                   << "missing electron, positron, and/or gamma particles "
                      "(required for "
                   << this->label() << ")");

    // Save particle properties
    host_data.electron_mass = particles.get(host_data.ids.electron).mass();

    // Load differential cross sections
    make_builder(&host_data.differential_xs.elements)
        .reserve(materials.num_elements());
    for (auto el_id : range(ElementId{materials.num_elements()}))
    {
        AtomicNumber z_number = materials.get(el_id).atomic_number();
        this->append_table(load_sb_table(z_number), &host_data.differential_xs);
    }
    CELER_ASSERT(host_data.differential_xs.elements.size()
                 == materials.num_elements());

    // Move to mirrored data, copying to device
    data_ = CollectionMirror<detail::SeltzerBergerData>{std::move(host_data)};

    CELER_ENSURE(this->data_);
}

//---------------------------------------------------------------------------//
/*!
 * Particle types and energy ranges that this model applies to.
 */
auto SeltzerBergerModel::applicability() const -> SetApplicability
{
    // TODO: Do we need to load applicabilities, e.g. lower and upper, from
    // tables?

    Applicability electron_applic;
    electron_applic.particle = this->host_pointers().ids.electron;
    electron_applic.lower    = zero_quantity();
    electron_applic.upper    = units::MevEnergy{1e8};

    Applicability positron_applic = electron_applic;
    positron_applic.particle      = this->host_pointers().ids.positron;

    return {electron_applic, positron_applic};
}

//---------------------------------------------------------------------------//
/*!
 * Apply the interaction kernel.
 */
void SeltzerBergerModel::interact(
    CELER_MAYBE_UNUSED const ModelInteractRefs<MemSpace::device>& pointers) const
{
#if CELERITAS_USE_CUDA
    detail::seltzer_berger_interact(this->device_pointers(), pointers);
#else
    CELER_ASSERT_UNREACHABLE();
#endif
}

//---------------------------------------------------------------------------//
/*!
 * Get the model ID for this model.
 */
ModelId SeltzerBergerModel::model_id() const
{
    return this->host_pointers().ids.model;
}

//---------------------------------------------------------------------------//
/*!
 * Construct differential cross section tables for a single element.
 *
 * Here, x = log of scaled incident energy (E / MeV)
 * and y = scaled exiting energy (E_gamma / E_inc)
 * and values are the cross sections.
 */
void SeltzerBergerModel::append_table(const ImportSBTable& imported,
                                      HostXsTables*        tables) const
{
    auto reals = make_builder(&tables->reals);

    CELER_ASSERT(!imported.value.empty()
                 && imported.value.size()
                        == imported.x.size() * imported.y.size());
    const size_type num_x = imported.x.size();
    const size_type num_y = imported.y.size();

    detail::SBElementTableData table;

    // TODO: we could probably use a single x and y grid for all elements.
    // Only Z = 100 has different energy grids.
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
        const real_type* iter = &tables->reals[table.grid.at(i, 0)];

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
} // namespace celeritas
