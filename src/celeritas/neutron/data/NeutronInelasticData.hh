//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/neutron/data/NeutronInelasticData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "corecel/grid/NonuniformGridData.hh"
#include "corecel/grid/TwodGridData.hh"
#include "corecel/math/Quantity.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Scalar data for neutron-nucleus inelastic interactions.
 */
struct NeutronInelasticScalars
{
    // Particle IDs
    ParticleId neutron_id;
    ParticleId proton_id;

    // Particle mass * c^2 [MeV]
    units::MevMass neutron_mass;
    units::MevMass proton_mass;

    //! Number of nucleon-nucleon channels
    static CELER_CONSTEXPR_FUNCTION size_type num_channels() { return 2; }

    //! Model's maximum energy limit [MeV]
    static CELER_CONSTEXPR_FUNCTION units::MevEnergy max_valid_energy()
    {
        // Below the pion production threshold
        return units::MevEnergy{320};
    }

    //! Whether data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return neutron_id && proton_id && neutron_mass > zero_quantity()
               && neutron_mass > zero_quantity();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Parameters of Stepanov's function to fit nucleon-nucleon cross sections
 * below 10 MeV.
 */
struct StepanovParameters
{
    real_type xs_zero;  //!< nucleon-nucleon cross section at the zero energy
    real_type slope;  //!< parameter used for the low energy threshold
    Real3 coeffs;  //!< coefficients of a second order Stepanov's function
};

//---------------------------------------------------------------------------//
/*!
 * Components of nuclear zone properties of the Bertini cascade model.
 */
struct ZoneComponent
{
    using NucleonArray = Array<real_type, 2>;  //!< [proton, neutron]

    real_type radius{};  //!< radius of zones in [femtometer]
    real_type volume{};  //!< volume of zones in [femtometer^3]
    NucleonArray density{0, 0};  //!< nucleon densities [1/femtometer^3]
    NucleonArray fermi_mom{0, 0};  //!< fermi momenta in [MeV/c]
    NucleonArray potential{0, 0};  //!< nucleon potentials [MeV]
};

//---------------------------------------------------------------------------//
/*!
 * Data characterizing the nuclear zones.
 */
struct NuclearZones
{
    ItemRange<ZoneComponent> zones;

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const { return !zones.empty(); }
};

//---------------------------------------------------------------------------//
/*!
 * Device data for nuclear zone properties
 */
template<Ownership W, MemSpace M>
struct NuclearZoneData
{
    template<class T>
    using Items = Collection<T, W, M>;
    template<class T>
    using IsotopeItems = Collection<T, W, M, IsotopeId>;

    //// MEMBER DATA ////

    // Nuclear zone data
    Items<ZoneComponent> components;
    IsotopeItems<NuclearZones> zones;

    //! Whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return !components.empty() && !zones.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    NuclearZoneData& operator=(NuclearZoneData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        components = other.components;
        zones = other.zones;

        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Device data for creating an interactor.
 */
template<Ownership W, MemSpace M>
struct NeutronInelasticData
{
    template<class T>
    using Items = Collection<T, W, M>;
    template<class T>
    using ElementItems = Collection<T, W, M, ElementId>;
    template<class T>
    using ChannelItems = Collection<T, W, M, ChannelId>;

    //// MEMBER DATA ////

    // Scalar data
    NeutronInelasticScalars scalars;

    // Microscopic (element) cross section data (G4PARTICLEXS/neutron/inelZ)
    ElementItems<NonuniformGridRecord> micro_xs;

    // Tabulated nucleon-nucleon cross section data
    ChannelItems<NonuniformGridRecord> nucleon_xs;

    // Parameters of necleon-nucleon cross sections below 10 MeV
    ChannelItems<StepanovParameters> xs_params;

    // Tabulated nucleon-nucleon angular distribution data
    ChannelItems<TwodGridData> angular_cdf;

    // Backend data
    Items<real_type> reals;

    // Nuclear zone data
    NuclearZoneData<W, M> nuclear_zones;

    //! Whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return scalars && !micro_xs.empty() && !nucleon_xs.empty()
               && !xs_params.empty() && !reals.empty() && nuclear_zones
               && !angular_cdf.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    NeutronInelasticData& operator=(NeutronInelasticData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        scalars = other.scalars;
        micro_xs = other.micro_xs;
        nucleon_xs = other.nucleon_xs;
        xs_params = other.xs_params;
        reals = other.reals;
        nuclear_zones = other.nuclear_zones;
        angular_cdf = other.angular_cdf;

        return *this;
    }
};

using NeutronInelasticHostRef = HostCRef<NeutronInelasticData>;
using NeutronInelasticDeviceRef = DeviceCRef<NeutronInelasticData>;
using NeutronInelasticRef = NativeCRef<NeutronInelasticData>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
