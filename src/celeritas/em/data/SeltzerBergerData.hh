//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/data/SeltzerBergerData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/Collection.hh"
#include "corecel/grid/TwodGridData.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"

#include "ElectronBremsData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Seltzer-Berger differential cross section tables for a single element.
 *
 * The 2D grid data is organized by \f$ \log E \f$ on the *x* axis and
 * fractional exiting energy (0 to 1) on the *y* axis. The values are in
 * millibarns, but their magnitude isn't important since we always take ratios.
 *
 * \c argmax is the y index of the largest cross section at a given incident
 * energy point.
 *
 * \todo We could use way smaller integers for argmax, even i/j here, because
 * these tables are so small.
 */
struct SBElementTableData
{
    using EnergyUnits = units::LogMev;
    using XsUnits = units::Millibarn;

    TwodGridData grid;  //!< Cross section grid and data
    ItemRange<size_type> argmax;  //!< Y index of the largest XS for each
                                  //!< energy

    explicit CELER_FUNCTION operator bool() const
    {
        return grid && argmax.size() == grid.x.size();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Bremsstrahlung differential cross section (DCS) data for SB sampling.
 *
 * The value grids are organized per element ID, and each 2D grid is:
 * - x: logarithm of the energy [MeV] of the incident charged dparticle
 * - y: ratio of exiting photon energy to incident particle energy
 * - value: differential cross section (microbarns)
 */
template<Ownership W, MemSpace M>
struct SeltzerBergerTableData
{
    //// MEMBER FUNCTIONS ////

    template<class T>
    using Items = Collection<T, W, M>;
    template<class T>
    using ElementItems = Collection<T, W, M, ElementId>;

    //// MEMBER DATA ////

    Items<real_type> reals;
    Items<size_type> sizes;
    ElementItems<SBElementTableData> elements;

    //// MEMBER FUNCTIONS ////

    //! Whether the data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return !reals.empty() && !sizes.empty() && !elements.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    SeltzerBergerTableData&
    operator=(SeltzerBergerTableData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        reals = other.reals;
        sizes = other.sizes;
        elements = other.elements;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Device data for sampling SeltzerBergerInteractor.
 */
template<Ownership W, MemSpace M>
struct SeltzerBergerData
{
    using MevMass = units::MevMass;
    using Energy = units::MevEnergy;

    //// MEMBER DATA ////

    //! IDs in a separate struct for readability/easier copying
    ElectronBremIds ids;

    //! Electron mass [MeV / c^2]
    MevMass electron_mass;

    //! High energy limit of the model
    Energy high_energy_limit;

    // Differential cross section storage
    SeltzerBergerTableData<W, M> differential_xs;

    //// MEMBER FUNCTIONS ////

    //! Whether the data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return ids && electron_mass > zero_quantity() && differential_xs;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    SeltzerBergerData& operator=(SeltzerBergerData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        ids = other.ids;
        electron_mass = other.electron_mass;
        high_energy_limit = other.high_energy_limit;
        differential_xs = other.differential_xs;
        return *this;
    }
};

using SeltzerBergerDeviceRef = DeviceCRef<SeltzerBergerData>;
using SeltzerBergerHostRef = HostCRef<SeltzerBergerData>;
using SeltzerBergerRef = NativeCRef<SeltzerBergerData>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
