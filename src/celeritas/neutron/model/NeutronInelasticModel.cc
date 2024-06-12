//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/neutron/model/NeutronInelasticModel.cc
//---------------------------------------------------------------------------//
#include "NeutronInelasticModel.hh"

#include "corecel/math/ArrayOperators.hh"
#include "corecel/math/Quantity.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/grid/GenericGridBuilder.hh"
#include "celeritas/io/ImportPhysicsVector.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/phys/InteractionApplier.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleParams.hh"

#include "CascadeOptions.hh"

#include "detail/NuclearZoneBuilder.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from model ID and other necessary data.
 */
NeutronInelasticModel::NeutronInelasticModel(ActionId id,
                                             ParticleParams const& particles,
                                             MaterialParams const& materials,
                                             CascadeOptions const& options,
                                             ReadData load_data)
{
    CELER_EXPECT(id);
    CELER_EXPECT(load_data);

    HostVal<NeutronInelasticData> data;

    // Save IDs
    data.scalars.action_id = id;
    data.scalars.neutron_id = particles.find(pdg::neutron());
    data.scalars.proton_id = particles.find(pdg::proton());

    CELER_VALIDATE(data.scalars.neutron_id && data.scalars.proton_id,
                   << "missing neutron and/or proton particles (required for "
                   << this->description() << ")");

    // Save particle properties
    data.scalars.neutron_mass = particles.get(data.scalars.neutron_id).mass();
    data.scalars.proton_mass = particles.get(data.scalars.proton_id).mass();
    CELER_EXPECT(data.scalars);

    // Load neutron inelastic cross section data
    CollectionBuilder micro_xs{&data.micro_xs};
    GenericGridBuilder build_grid{&data.reals};
    for (auto el_id : range(ElementId{materials.num_elements()}))
    {
        AtomicNumber z = materials.get(el_id).atomic_number();
        micro_xs.push_back(build_grid(load_data(z)));
    }
    CELER_ASSERT(data.micro_xs.size() == materials.num_elements());

    // Build nucleon-nucleon cross section data
    size_type num_channels = data.scalars.num_channels();
    make_builder(&data.nucleon_xs).reserve(num_channels);
    auto xs_params = make_builder(&data.xs_params);
    xs_params.reserve(num_channels);

    auto bins = this->get_channel_bins();
    for (auto channel_id : range(ChannelId{num_channels}))
    {
        // Add nucleon-nucleon cross section parameters and data
        ChannelXsData const& channel_data = this->get_channel_xs(channel_id);
        CELER_ASSERT(channel_data.par.slope > 0);
        xs_params.push_back(channel_data.par);

        GenericGridBuilder build_grid{&data.reals};
        make_builder(&data.nucleon_xs)
            .push_back(build_grid(bins, make_span(channel_data.xs)));
    }
    CELER_ASSERT(data.nucleon_xs.size() == num_channels);
    CELER_ASSERT(data.xs_params.size() == data.nucleon_xs.size());

    // Build (A, Z)-dependent nuclear zone data
    detail::NuclearZoneBuilder zone_builder(
        options, data.scalars, &data.nuclear_zones);

    for (auto iso_id : range(IsotopeId{materials.num_isotopes()}))
    {
        zone_builder(materials.get(iso_id));
    }
    CELER_ASSERT(data.nuclear_zones.zones.size() == materials.num_isotopes());

    // Move to mirrored data, copying to device
    data_ = CollectionMirror<NeutronInelasticData>{std::move(data)};
    CELER_ENSURE(this->data_);
}

//---------------------------------------------------------------------------//
/*!
 * Particle types and energy ranges that this model applies to.
 */
auto NeutronInelasticModel::applicability() const -> SetApplicability
{
    Applicability neutron_applic;
    neutron_applic.particle = this->host_ref().scalars.neutron_id;
    neutron_applic.lower = zero_quantity();
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
 * Get the particle-nucleon cross section (in barn) as a function of particle
 * energy. Only neutron-neutron, neutron-proton and proton-proton channels are
 * tabulated in [10, 320] (MeV) where pion production is not likely. The cross
 * sections below 10 MeV will be calculated on the fly using the Stepanov's
 * function. Tabulated data of cross sections and parameters at the low energy
 * are from G4CascadePPChannel, G4CascadeNPChannel and G4CascadeNNChannel of
 * the Geant4 11.2 release. Also note that the channel cross sections of
 * nucleon-nucleon are same as their total cross sections in the energy range.
 *
 * H. W. Bertini, "Low-Energy Intranuclear Cascade Calculation", Phys. Rev.
 * Vol. 131, page 1801 (1963). W. Hess, "Summary of High-Energy Nucleon-
 * Nucleon Cross-Section Data", Rev. Mod. Phys. Vol. 30, page 368 (1958).
 */
auto NeutronInelasticModel::get_channel_xs(ChannelId ch_id)
    -> ChannelXsData const&
{
    CELER_EXPECT(ch_id);
    static ChannelXsData const channels[]
        = {{{17.613, 4.00, {0.0069466, 9.0692, -5.0574}},
            {0.8633,
             0.6746,
             0.4952,
             0.3760,
             0.2854,
             0.2058,
             0.1357,
             0.0937,
             0.0691,
             0.0552,
             0.0445,
             0.0388,
             0.0351}},
           {{20.360, 1.92, {0.0053107, 3.0885, -1.1748}},
            {0.3024,
             0.2359,
             0.1733,
             0.1320,
             0.1007,
             0.0749,
             0.0519,
             0.0388,
             0.0316,
             0.0278,
             0.0252,
             0.0240,
             0.0233}},
           {{17.613, 4.00, {0.0069466, 9.0692, -5.0574}},
            {0.8633,
             0.6746,
             0.4952,
             0.3760,
             0.2854,
             0.2058,
             0.1357,
             0.0937,
             0.0691,
             0.0552,
             0.0445,
             0.0388,
             0.0351}}};
    return channels[ch_id.unchecked_get()];
}

//---------------------------------------------------------------------------//
/*!
 * Get the energy bins (MeV) of nucleon-nucleon channel data in [10, 320] (MeV)
 * from the G4PionNucSampler class. Note that the GeV unit is used in the
 * Bertini cascade G4NucleiModel class.
 */
Span<double const> NeutronInelasticModel::get_channel_bins() const
{
    static ChannelArray const bins
        = {10, 13, 18, 24, 32, 42, 56, 75, 100, 130, 180, 240, 320};

    return make_span(bins);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
