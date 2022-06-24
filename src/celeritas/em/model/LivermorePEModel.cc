//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/LivermorePEModel.cc
//---------------------------------------------------------------------------//
#include "LivermorePEModel.hh"

#include <algorithm>
#include <utility>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/cont/Array.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/ScopedTimeLog.hh"
#include "corecel/math/Quantity.hh"
#include "celeritas/Types.hh"
#include "celeritas/em/data/LivermorePEData.hh"
#include "celeritas/em/generated/LivermorePEInteract.hh"
#include "celeritas/grid/XsGridData.hh"
#include "celeritas/mat/ElementView.hh"
#include "celeritas/phys/Applicability.hh"
#include "celeritas/phys/PDGNumber.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from model ID and other necessary data.
 */
LivermorePEModel::LivermorePEModel(ActionId              id,
                                   const ParticleParams& particles,
                                   const MaterialParams& materials,
                                   ReadData              load_data)
{
    CELER_EXPECT(id);
    CELER_EXPECT(load_data);

    LivermorePEData<Ownership::value, MemSpace::host> host_data;

    // Save IDs
    host_data.ids.action   = id;
    host_data.ids.electron = particles.find(pdg::electron());
    host_data.ids.gamma    = particles.find(pdg::gamma());
    CELER_VALIDATE(host_data.ids,
                   << "missing electron and/or gamma particles "
                      "(required for "
                   << this->description() << ")");

    // Save particle properties
    host_data.inv_electron_mass
        = 1 / particles.get(host_data.ids.electron).mass().value();

    // Load Livermore cross section data
    CELER_LOG(status) << "Reading and building Livermore PE model data";
    ScopedTimeLog scoped_time;
    make_builder(&host_data.xs.elements).reserve(materials.num_elements());
    for (auto el_id : range(ElementId{materials.num_elements()}))
    {
        AtomicNumber z = materials.get(el_id).atomic_number();
        this->append_element(load_data(z), &host_data.xs);
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
    photon_applic.lower    = zero_quantity();
    photon_applic.upper    = max_quantity();

    return {photon_applic};
}

//---------------------------------------------------------------------------//
/*!
 * Get the microscopic cross sections for the given particle and material.
 */
auto LivermorePEModel::micro_xs(Applicability) const -> MicroXsBuilders
{
    // Cross sections are calculated on the fly
    return {};
}

//---------------------------------------------------------------------------//
//!@{
/*!
 * Apply the interaction kernel.
 */
void LivermorePEModel::execute(CoreDeviceRef const& data) const
{
    generated::livermore_pe_interact(this->device_ref(), data);
}

void LivermorePEModel::execute(CoreHostRef const& data) const
{
    generated::livermore_pe_interact(this->host_ref(), data);
}

//!@}
//---------------------------------------------------------------------------//
/*!
 * Get the model ID for this model.
 */
ActionId LivermorePEModel::action_id() const
{
    return this->host_ref().ids.action;
}

//---------------------------------------------------------------------------//
/*!
 * Construct cross section data for a single element.
 */
void LivermorePEModel::append_element(const ImportLivermorePE& inp,
                                      HostXsData*              xs) const
{
    CELER_EXPECT(!inp.shells.empty());
    if (CELERITAS_DEBUG)
    {
        CELER_EXPECT(inp.thresh_lo <= inp.thresh_hi);
        for (const auto& shell : inp.shells)
        {
            CELER_EXPECT(shell.param_lo.size() == 6);
            CELER_EXPECT(shell.param_hi.size() == 6);
            CELER_EXPECT(shell.binding_energy <= inp.thresh_lo);
        }
    }

    auto reals = make_builder(&xs->reals);

    LivermoreElement el;

    // Add tabulated total cross sections
    el.xs_lo.grid  = reals.insert_back(inp.xs_lo.x.begin(), inp.xs_lo.x.end());
    el.xs_lo.value = reals.insert_back(inp.xs_lo.y.begin(), inp.xs_lo.y.end());
    el.xs_lo.grid_interp  = Interp::linear;
    el.xs_lo.value_interp = Interp::linear;
    el.xs_hi.grid  = reals.insert_back(inp.xs_hi.x.begin(), inp.xs_hi.x.end());
    el.xs_hi.value = reals.insert_back(inp.xs_hi.y.begin(), inp.xs_hi.y.end());
    el.xs_hi.grid_interp  = Interp::linear;
    el.xs_hi.value_interp = Interp::linear; // TODO: spline

    // Add energy thresholds for using low and high xs parameterization
    el.thresh_lo = MevEnergy{inp.thresh_lo};
    el.thresh_hi = MevEnergy{inp.thresh_hi};

    // Allocate subshell data
    std::vector<LivermoreSubshell> shells(inp.shells.size());

    // Add subshell data
    for (auto i : range(inp.shells.size()))
    {
        // Ionization energy
        shells[i].binding_energy = MevEnergy{inp.shells[i].binding_energy};

        // Tabulated subshell cross section
        shells[i].xs.grid  = reals.insert_back(inp.shells[i].energy.begin(),
                                              inp.shells[i].energy.end());
        shells[i].xs.value = reals.insert_back(inp.shells[i].xs.begin(),
                                               inp.shells[i].xs.end());
        shells[i].xs.grid_interp  = Interp::linear;
        shells[i].xs.value_interp = Interp::linear;

        // Subshell cross section fit parameters
        std::copy(inp.shells[i].param_lo.begin(),
                  inp.shells[i].param_lo.end(),
                  shells[i].param[0].begin());
        std::copy(inp.shells[i].param_hi.begin(),
                  inp.shells[i].param_hi.end(),
                  shells[i].param[1].begin());

        CELER_ASSERT(shells[i]);
    }
    el.shells
        = make_builder(&xs->shells).insert_back(shells.begin(), shells.end());

    // Add the elemental data
    CELER_ASSERT(el);
    make_builder(&xs->elements).push_back(el);

    CELER_ENSURE(el.shells.size() == inp.shells.size());
}

//---------------------------------------------------------------------------//
} // namespace celeritas
