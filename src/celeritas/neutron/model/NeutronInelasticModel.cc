//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/neutron/model/NeutronInelasticModel.cc
//---------------------------------------------------------------------------//
#include "NeutronInelasticModel.hh"

#include "corecel/math/ArrayOperators.hh"
#include "corecel/math/Quantity.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/grid/NonuniformGridBuilder.hh"
#include "celeritas/grid/TwodGridBuilder.hh"
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
    : StaticConcreteAction(id,
                           "neutron-inelastic-bertini",
                           "interact by neutron inelastic (Bertini)")
{
    CELER_EXPECT(id);
    CELER_EXPECT(load_data);

    HostVal<NeutronInelasticData> data;

    // Save IDs
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
    NonuniformGridBuilder build_grid{&data.reals};
    for (auto el_id : range(ElementId{materials.num_elements()}))
    {
        AtomicNumber z = materials.get(el_id).atomic_number();
        micro_xs.push_back(build_grid(load_data(z)));
    }
    CELER_ASSERT(data.micro_xs.size() == materials.num_elements());

    // Build nucleon-nucleon cross section and angular distribution data
    size_type num_channels = data.scalars.num_channels();
    make_builder(&data.nucleon_xs).reserve(num_channels);
    auto xs_params = make_builder(&data.xs_params);
    xs_params.reserve(num_channels);

    auto xs_energy_bins = this->get_xs_energy_bins();
    auto cdf_energy_bins = this->get_cdf_energy_bins();
    auto cos_theta_bins = this->get_cos_theta_bins();

    NonuniformGridBuilder build_xs_grid{&data.reals};
    TwodGridBuilder build_cdf_grid{&data.reals};
    CollectionBuilder nucleon_xs{&data.nucleon_xs};
    CollectionBuilder angular_cdf{&data.angular_cdf};
    for (auto channel_id : range(ChannelId{num_channels}))
    {
        // Add nucleon-nucleon cross section parameters and data
        ChannelData const& channel_data = this->get_channel_data(channel_id);
        CELER_ASSERT(channel_data.par.slope > 0);
        xs_params.push_back(channel_data.par);
        nucleon_xs.push_back(
            build_xs_grid(xs_energy_bins, make_span(channel_data.xs)));

        // Add nucleon-nucleon two-body angular distribution data
        angular_cdf.push_back(build_cdf_grid(
            cdf_energy_bins, cos_theta_bins, make_span(channel_data.cdf)));
    }
    CELER_ASSERT(data.nucleon_xs.size() == num_channels);
    CELER_ASSERT(data.angular_cdf.size() == num_channels);
    CELER_ASSERT(data.xs_params.size() == data.nucleon_xs.size());

    // Build (A, Z)-dependent nuclear zone data
    detail::NuclearZoneBuilder build_nuclear_zones(
        options, data.scalars, &data.nuclear_zones);

    for (auto iso_id : range(IsotopeId{materials.num_isotopes()}))
    {
        build_nuclear_zones(materials.get(iso_id));
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
void NeutronInelasticModel::step(CoreParams const&, CoreStateHost&) const
{
    CELER_NOT_IMPLEMENTED("Neutron inelastic interaction");
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void NeutronInelasticModel::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
/*!
 * Get the particle-nucleon cross section (in barn) and the c.d.f (cumulative
 * distribution function) of cos theta distribution as a function of particle
 * energy.
 */
auto NeutronInelasticModel::get_channel_data(ChannelId ch_id)
    -> ChannelData const&
{
    CELER_EXPECT(ch_id);
    static ChannelData const channels[]
        = {{{17.613, 4.00, {0.0069466, 9.0692, -5.0574}},  // neutron-neutron
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
             0.0351},
            {0.0000, 0.0075, 0.0300, 0.0670, 0.1170, 0.1785, 0.2500, 0.3290,
             0.4130, 0.5000, 0.5870, 0.6710, 0.7500, 0.8215, 0.8830, 0.9330,
             0.9700, 0.9925, 1.0000, 0.0000, 0.0095, 0.0361, 0.0766, 0.1284,
             0.1902, 0.2605, 0.3370, 0.4174, 0.5000, 0.5826, 0.6630, 0.7395,
             0.8098, 0.8716, 0.9234, 0.9638, 0.9905, 1.0000, 0.0000, 0.0104,
             0.0388, 0.0808, 0.1334, 0.1954, 0.2652, 0.3405, 0.4193, 0.5000,
             0.5807, 0.6595, 0.7348, 0.8046, 0.8666, 0.9192, 0.9611, 0.9896,
             1.0000, 0.0000, 0.0102, 0.0374, 0.0776, 0.1290, 0.1906, 0.2610,
             0.3375, 0.4177, 0.5000, 0.5823, 0.6625, 0.7390, 0.8094, 0.8710,
             0.9224, 0.9626, 0.9898, 1.0000, 0.0000, 0.0099, 0.0353, 0.0730,
             0.1227, 0.1837, 0.2549, 0.3331, 0.4154, 0.5000, 0.5846, 0.6669,
             0.7451, 0.8163, 0.8773, 0.9270, 0.9647, 0.9901, 1.0000, 0.0000,
             0.0102, 0.0364, 0.0750, 0.1255, 0.1869, 0.2580, 0.3355, 0.4167,
             0.5000, 0.5833, 0.6645, 0.7420, 0.8131, 0.8745, 0.9250, 0.9636,
             0.9898, 1.0000}},
           {{20.360, 1.92, {0.0053107, 3.0885, -1.1748}},  // neutron-proton
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
             0.0233},
            {0.0000, 0.0075, 0.0300, 0.0670, 0.1170, 0.1785, 0.2500, 0.3290,
             0.4130, 0.5000, 0.5870, 0.6710, 0.7500, 0.8215, 0.8830, 0.9330,
             0.9700, 0.9925, 1.0000, 0.0000, 0.0149, 0.0569, 0.1182, 0.1889,
             0.2613, 0.3320, 0.3995, 0.4642, 0.5264, 0.5858, 0.6428, 0.6998,
             0.7596, 0.8229, 0.8872, 0.9450, 0.9855, 1.0000, 0.0000, 0.0180,
             0.0681, 0.1387, 0.2161, 0.2909, 0.3604, 0.4252, 0.4877, 0.5485,
             0.6063, 0.6599, 0.7113, 0.7645, 0.8225, 0.8844, 0.9426, 0.9847,
             1.0000, 0.0000, 0.0235, 0.0876, 0.1746, 0.2638, 0.3428, 0.4101,
             0.4702, 0.5288, 0.5873, 0.6421, 0.6897, 0.7313, 0.7731, 0.8219,
             0.8795, 0.9384, 0.9833, 1.0000, 0.0000, 0.0193, 0.0722, 0.1447,
             0.2200, 0.2874, 0.3448, 0.3965, 0.4488, 0.5062, 0.5685, 0.6331,
             0.6983, 0.7637, 0.8290, 0.8923, 0.9478, 0.9863, 1.0000, 0.0000,
             0.0201, 0.0745, 0.1472, 0.2208, 0.2857, 0.3413, 0.3918, 0.4424,
             0.4971, 0.5569, 0.6205, 0.6864, 0.7531, 0.8197, 0.8849, 0.9434,
             0.9850, 1.0000}}};

    return channels[ch_id.unchecked_get()];
}

//---------------------------------------------------------------------------//
/*!
 * Get the energy bins (MeV) of nucleon-nucleon channel data in [10, 320] (MeV)
 * from the G4PionNucSampler class. Note that the GeV unit is used in the
 * Bertini cascade G4NucleiModel class.
 */
Span<double const> NeutronInelasticModel::get_xs_energy_bins() const
{
    static Array<double, 13> const bins
        = {10, 13, 18, 24, 32, 42, 56, 75, 100, 130, 180, 240, 320};

    return make_span(bins);
}

//---------------------------------------------------------------------------//
/*!
 * Get the energy bins of the two-body nucleon-nucleon angular distribution
 * data in [0, 320] (MeV) from G4PP2PPAngDst and G4NP2NPAngDst classes. The
 * number of bins and c.d.f values of the angular probability are reorganized
 * with a common data structure.
 */
Span<double const> NeutronInelasticModel::get_cdf_energy_bins() const
{
    static Array<double, 6> const bins = {0, 90, 130, 200, 300, 320};

    return make_span(bins);
}

//---------------------------------------------------------------------------//
/*!
 * Get the cosine angular bins of the nucleon-nucleon angular distribution data
 * in [-1, 1] from G4PP2PPAngDst and G4NP2NPAngDst classes.
 */
Span<double const> NeutronInelasticModel::get_cos_theta_bins() const
{
    // clang-format off
    static Array<double, 19> const bins
        = {-1.000, -0.985, -0.940, -0.866, -0.766, -0.643, -0.500,
           -0.342, -0.174,  0.000,  0.174,  0.342,  0.500,  0.643,
            0.766,  0.866,  0.940,  0.985, 1.000};
    // clang-format on

    return make_span(bins);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
