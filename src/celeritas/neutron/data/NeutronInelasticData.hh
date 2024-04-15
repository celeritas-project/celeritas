//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/neutron/data/NeutronInelasticData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "corecel/math/Quantity.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/grid/GenericGridData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Scalar data for neutron-nucleus inelastic interactions.
 */
struct NeutronInelasticScalars
{
    // Action and particle IDs
    ActionId action_id;
    ParticleId neutron_id;

    // Particle mass * c^2 [MeV]
    units::MevMass neutron_mass;

    //! Number of nucleon-nucleon channels
    static CELER_CONSTEXPR_FUNCTION size_type num_channels() { return 3; }

    //! Model's maximum energy limit [MeV]
    static CELER_CONSTEXPR_FUNCTION units::MevEnergy max_valid_energy()
    {
        // Below the pion production threshold
        return units::MevEnergy{320};
    }

    //! Whether data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return action_id && neutron_id && neutron_mass > zero_quantity();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Parameters of Stepanov's function to fit nucleon-nucleon cross sections
 * below 10 MeV.
 */
struct StepanovParameters
{
    real_type xs_o;  //!< nucleon-nucleon cross section at the zero energy
    real_type slope;  //!< parameter used for the low energy threshold
    Real3 coeffs;  //!< coefficients of a second order Stepanov's function
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
    ElementItems<GenericGridData> micro_xs;

    // Tabulated nucleon-nucleon cross section data
    ChannelItems<GenericGridData> nucleon_xs;

    // Parameters of necleon-nucleon cross sections below 10 MeV
    ChannelItems<StepanovParameters> xs_params;

    // Backend data
    Items<real_type> reals;

    //! Whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return scalars && !micro_xs.empty() && !nucleon_xs.empty()
               && !xs_params.empty() && !reals.empty();
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
        return *this;
    }
};

using NeutronInelasticHostRef = HostCRef<NeutronInelasticData>;
using NeutronInelasticDeviceRef = DeviceCRef<NeutronInelasticData>;
using NeutronInelasticRef = NativeCRef<NeutronInelasticData>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
