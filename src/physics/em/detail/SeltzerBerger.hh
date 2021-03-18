//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SeltzerBerger.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Types.hh"
#include "physics/base/Types.hh"
#include "physics/base/Units.hh"
#include "physics/grid/TwodGridInterface.hh"
#include "physics/material/Types.hh"

namespace celeritas
{
struct ModelInteractPointers;

namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Seltzer-Berger differential cross section tables for a single element.
 *
 * The 2D grid data is organized by log E on the x axis and fractional exiting
 * energy (0 to 1) on the y axis. The values are in millibarns, but their
 * magnitude isn't important since we always take ratios.
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
    using XsUnits     = units::Millibarn;

    TwodGridData         grid;   //!< Cross section grid and data
    ItemRange<size_type> argmax; //!< Y index of the largest XS for each energy

    explicit inline CELER_FUNCTION operator bool() const
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
    using Data = Collection<T, W, M>;
    template<class T>
    using ElementData = Collection<T, W, M, ElementId>;

    //// MEMBER DATA ////

    Data<real_type>                 reals;
    Data<size_type>                 sizes;
    ElementData<SBElementTableData> elements;

    //// MEMBER FUNCTIONS ////

    //! Whether the data is assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        return !reals.empty() && !sizes.empty() && !elements.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    SeltzerBergerTableData&
    operator=(const SeltzerBergerTableData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        reals    = other.reals;
        sizes    = other.sizes;
        elements = other.elements;
        return *this;
    }
};

//! Helper struct for making assignment easier
struct SeltzerBergerIds
{
    //! Model ID
    ModelId model;
    //! ID of an electron
    ParticleId electron;
    //! ID of an positron
    ParticleId positron;
    //! ID of a gamma
    ParticleId gamma;

    //! Whether the IDs are assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        return model && electron && positron && gamma;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Device data for sampling SeltzerBergerInteractor.
 */
template<Ownership W, MemSpace M>
struct SeltzerBergerData
{
    //// MEMBER DATA ////

    //! IDs in a separate struct for readability/easier copying
    SeltzerBergerIds ids;

    //! Electron mass [MevMass]
    real_type electron_mass;

    // Differential cross section storage
    SeltzerBergerTableData<W, M> differential_xs;

    //// MEMBER FUNCTIONS ////

    //! Whether the data is assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        return ids && electron_mass > 0 && differential_xs;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    SeltzerBergerData& operator=(const SeltzerBergerData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        ids             = other.ids;
        electron_mass   = other.electron_mass;
        differential_xs = other.differential_xs;
        return *this;
    }
};

using SeltzerBergerDeviceRef
    = SeltzerBergerData<Ownership::const_reference, MemSpace::device>;
using SeltzerBergerHostRef
    = SeltzerBergerData<Ownership::const_reference, MemSpace::host>;

//---------------------------------------------------------------------------//
// KERNEL LAUNCHERS
//---------------------------------------------------------------------------//

// Launch the Seltzer-Berger interaction
void seltzer_berger_interact(const SeltzerBergerDeviceRef& device_pointers,
                             const ModelInteractPointers&  interaction);

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
