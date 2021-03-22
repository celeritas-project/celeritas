//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LivermorePEModel.cc
//---------------------------------------------------------------------------//
#include "LivermorePEModel.hh"

#include "base/Assert.hh"
#include "base/CollectionBuilder.hh"
#include "comm/Device.hh"
#include "physics/base/PDGNumber.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from model ID and other necessary data.
 */
LivermorePEModel::LivermorePEModel(ModelId               id,
                                   const ParticleParams& particles,
                                   const MaterialParams& materials,
                                   ReadData              load_data,
                                   SPConstAtomicRelax    atomic_relaxation,
                                   size_type             num_vacancies)
{
    CELER_EXPECT(id);
    CELER_EXPECT(load_data);

    detail::LivermorePEData<Ownership::value, MemSpace::host> host_data;

    // Save IDs
    host_data.ids.model    = id;
    host_data.ids.electron = particles.find(pdg::electron());
    host_data.ids.gamma    = particles.find(pdg::gamma());
    CELER_VALIDATE(host_data.ids,
                   "Electron and gamma particles must be enabled to use the "
                   "Livermore Photoelectric Model.");

    // Save particle properties
    host_data.inv_electron_mass
        = 1 / particles.get(host_data.ids.electron).mass().value();

    // Load Livermore cross section data
    make_builder(&host_data.xs_data.elements).reserve(materials.num_elements());
    for (auto el_id : range(ElementId{materials.num_elements()}))
    {
        AtomicNumber z = materials.get(el_id).atomic_number();
        this->append_element(load_data(z), &host_data.xs_data);
    }
    CELER_ASSERT(host_data.xs_data.elements.size() == materials.num_elements());

    // Add atomic relaxation data
    if (atomic_relaxation)
    {
        CELER_ASSERT(num_vacancies > 0);
        resize(&relax_scratch_.vacancies, num_vacancies);
        relax_scratch_ref_          = relax_scratch_;
        host_data.atomic_relaxation = atomic_relaxation->device_pointers();
    }

    // Move to mirrored data, copying to device
    data_ = CollectionMirror<detail::LivermorePEData>{std::move(host_data)};
    CELER_ENSURE(this->data_);
}

//---------------------------------------------------------------------------//
/*!
 * Particle types and energy ranges that this model applies to.
 */
auto LivermorePEModel::applicability() const -> SetApplicability
{
    Applicability photon_applic;
    photon_applic.particle = this->host_pointers().ids.gamma;
    photon_applic.lower    = zero_quantity();
    photon_applic.upper    = max_quantity();

    return {photon_applic};
}

//---------------------------------------------------------------------------//
/*!
 * Apply the interaction kernel.
 */
void LivermorePEModel::interact(
    CELER_MAYBE_UNUSED const ModelInteractPointers& pointers) const
{
#if CELERITAS_USE_CUDA
    detail::livermore_pe_interact(
        this->device_pointers(), relax_scratch_ref_, pointers);
#else
    CELER_ASSERT_UNREACHABLE();
#endif
}

//---------------------------------------------------------------------------//
/*!
 * Get the model ID for this model.
 */
ModelId LivermorePEModel::model_id() const
{
    return this->host_pointers().ids.model;
}

//---------------------------------------------------------------------------//
/*!
 * Construct cross section data for a single element.
 */
void LivermorePEModel::append_element(const ImportLivermorePE& inp,
                                      HostXsData*              xs_data) const
{
    CELER_EXPECT(!inp.shells.empty());

    auto reals = make_builder(&xs_data->reals);

    detail::LivermoreElement el;

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
    std::vector<detail::LivermoreSubshell> shells(inp.shells.size());

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
        shells[i].param_lo = reals.insert_back(inp.shells[i].param_lo.begin(),
                                               inp.shells[i].param_lo.end());
        shells[i].param_hi = reals.insert_back(inp.shells[i].param_hi.begin(),
                                               inp.shells[i].param_hi.end());
    }
    el.shells = make_builder(&xs_data->shells)
                    .insert_back(shells.begin(), shells.end());

    // Add the elemental data
    make_builder(&xs_data->elements).push_back(el);

    CELER_ENSURE(el.shells.size() == inp.shells.size());
}

//---------------------------------------------------------------------------//
} // namespace celeritas
