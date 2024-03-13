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
 * Model and particles IDs for neutron--nucleus inelastic interactions.
 */
struct NeutronInelasticIds
{
    ActionId action;
    ParticleId neutron;

    //! Whether the IDs are assigned
    explicit CELER_FUNCTION operator bool() const { return action && neutron; }
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
    using IsotopeItems = Collection<T, W, M, IsotopeId>;

    //// MEMBER DATA ////

    //! Particle IDs
    NeutronInelasticIds ids;

    //! Particle mass * c^2 [MeV]
    units::MevMass neutron_mass;

    //! Microscopic (element) cross section data (G4PARTICLEXS/neutron/inelZ)
    Items<real_type> reals;
    ElementItems<GenericGridData> micro_xs;

    //! Model's minimum and maximum energy limit [MeV]
    static CELER_CONSTEXPR_FUNCTION units::MevEnergy min_valid_energy()
    {
        return units::MevEnergy{1e-7};
    }

    static CELER_CONSTEXPR_FUNCTION units::MevEnergy max_valid_energy()
    {
        return units::MevEnergy{2e+4};
    }

    //! Whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return ids && neutron_mass > zero_quantity() && !reals.empty()
               && !micro_xs.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    NeutronInelasticData& operator=(NeutronInelasticData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        ids = other.ids;
        neutron_mass = other.neutron_mass;
        reals = other.reals;
        micro_xs = other.micro_xs;
        return *this;
    }
};

using NeutronInelasticHostRef = HostCRef<NeutronInelasticData>;
using NeutronInelasticDeviceRef = DeviceCRef<NeutronInelasticData>;
using NeutronInelasticRef = NativeCRef<NeutronInelasticData>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
