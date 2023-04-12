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
#include "corecel/io/Logger.hh"
#include "corecel/io/ScopedTimeLog.hh"
#include "corecel/sys/ScopedMem.hh"
#include "celeritas/em/data/ElectronBremsData.hh"
#include "celeritas/em/generated/SeltzerBergerInteract.hh"
#include "celeritas/em/interactor/detail/PhysicsConstants.hh"
#include "celeritas/em/interactor/detail/SBPositronXsCorrector.hh"
#include "celeritas/io/ImportProcess.hh"
#include "celeritas/mat/MaterialParams.hh"
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
    CELER_LOG(status) << "Reading and building Seltzer Berger model data";
    ScopedTimeLog scoped_time;
    make_builder(&host_data.differential_xs.elements)
        .reserve(materials.num_elements());
    for (auto el_id : range(ElementId{materials.num_elements()}))
    {
        auto element = materials.get(el_id);
        this->append_table(element,
                           load_sb_table(element.atomic_number()),
                           &host_data.differential_xs,
                           host_data.electron_mass);
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
//!@{
/*!
 * Apply the interaction kernel.
 */
void SeltzerBergerModel::execute(CoreDeviceRef const& data) const
{
    generated::seltzer_berger_interact(this->device_ref(), data);
}

void SeltzerBergerModel::execute(CoreHostRef const& data) const
{
    generated::seltzer_berger_interact(this->host_ref(), data);
}
//!@}
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
void SeltzerBergerModel::append_table(ElementView const& element,
                                      ImportSBTable const& imported,
                                      HostXsTables* tables,
                                      Mass electron_mass) const
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

        if constexpr (CELERITAS_DEBUG)
        {
            using Energy = units::MevEnergy;

            // Check that the maximum scaled positron cross section is always
            // at the first reduced photon energy grid point
            real_type inc_energy = std::exp(imported.x[i]);
            SBPositronXsCorrector scale_xs(electron_mass,
                                           element,
                                           Energy{imported.y[0] * inc_energy},
                                           Energy{inc_energy});

            // When the reduced photon energy is 1 the scaling factor is 0
            size_type num_scaled = num_y - 1;
            CELER_ASSERT(imported.y[num_scaled] == 1);

            std::vector<real_type> scaled_xs(iter, iter + num_scaled);
            for (size_type j : range(num_scaled))
            {
                scaled_xs[j] *= scale_xs(Energy{imported.y[j] * inc_energy});
            }
            CELER_ASSERT(std::max_element(scaled_xs.begin(), scaled_xs.end())
                             - scaled_xs.begin()
                         == 0);
        }
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
