//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/neutron/data/NeutronCaptureData.hh
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
 * Model and particles IDs for the neutron capture process.
 */
struct NeutronCaptureIds
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
struct NeutronCaptureData
{
    template<class T>
    using Items = Collection<T, W, M>;
    template<class T>
    using ElementItems = Collection<T, W, M, ElementId>;
    template<class T>
    using IsotopeItems = Collection<T, W, M, IsotopeId>;

    //// MEMBER DATA ////

    //! Model and particle IDs
    NeutronCaptureIds ids;

    //! Particle mass * c^2 [MeV]
    units::MevMass neutron_mass;

    //! Microscopic (element) cross section data (G4PARTICLEXS/neutron/capZ)
    Items<real_type> reals;
    ElementItems<GenericGridData> micro_xs;

    //! Model's minimum and maximum energy limit [MeV]
    static CELER_CONSTEXPR_FUNCTION units::MevEnergy min_valid_energy()
    {
        return units::MevEnergy{1e-6};
    }

    static CELER_CONSTEXPR_FUNCTION units::MevEnergy max_valid_energy()
    {
        return units::MevEnergy{20};
    }

    //! Whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return ids && neutron_mass > zero_quantity() && !reals.empty()
               && !micro_xs.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    NeutronCaptureData& operator=(NeutronCaptureData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        ids = other.ids;
        neutron_mass = other.neutron_mass;
        reals = other.reals;
        micro_xs = other.micro_xs;
        return *this;
    }
};

using NeutronCaptureHostRef = HostCRef<NeutronCaptureData>;
using NeutronCaptureDeviceRef = DeviceCRef<NeutronCaptureData>;
using NeutronCaptureRef = NativeCRef<NeutronCaptureData>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
