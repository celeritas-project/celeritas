//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LivermorePE.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/StackAllocatorInterface.hh"
#include "base/Span.hh"
#include "base/Types.hh"
#include "physics/base/Types.hh"
#include "physics/base/Units.hh"
#include "physics/em/AtomicRelaxationInterface.hh"
#include "physics/grid/XsGridInterface.hh"
#include "physics/material/Types.hh"

namespace celeritas
{
template<MemSpace M>
struct ModelInteractRefs;

namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Electron subshell data.
 *
 * The binding energy of consecutive shells is *not* always decreasing.
 * However, it is guaranteed to be less than or equal to the parent element's
 * \c thresh_lo value.
 */
struct LivermoreSubshell
{
    using EnergyUnits = units::Mev;
    using XsUnits     = units::Barn;
    using Energy      = Quantity<EnergyUnits>;
    using Real6       = Array<real_type, 6>;

    // Binding energy of the electron
    Energy binding_energy;

    // Tabulated subshell photoionization cross section (used below 5 keV)
    GenericGridData xs;

    // Fit parameters for the integrated subshell photoionization cross
    // sections in the two different energy ranges (used above 5 keV)
    Array<Real6, 2> param;

    //! True if assigned and valid
    explicit inline CELER_FUNCTION operator bool() const
    {
        return binding_energy > celeritas::zero_quantity() && xs;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Elemental photoelectric cross sections for the Livermore model.
 */
struct LivermoreElement
{
    using Energy = LivermoreSubshell::Energy;

    // TOTAL CROSS SECTIONS

    // Total cross section below the K-shell energy. Uses linear interpolation.
    GenericGridData xs_lo;

    // Total cross section above the K-shell energy but below the energy
    // threshold for the parameterized cross sections. Uses spline
    // interpolation.
    GenericGridData xs_hi;

    // SUBSHELL CROSS SECTIONS

    ItemRange<LivermoreSubshell> shells;

    // Energy threshold for using the parameterized subshell cross sections in
    // the lower and upper energy range
    Energy thresh_lo; //!< Use tabulated XS below this energy
    Energy thresh_hi; //!< Use lower parameterization below, upper above

    //! True if assigned and valid
    explicit inline CELER_FUNCTION operator bool() const
    {
        // Note: xs_lo is not present for elements with only one subshell, so
        // it's valid for xs_lo to be unassigned.
        return xs_hi && !shells.empty() && thresh_lo <= thresh_hi;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Livermore photoelectric cross section data and binding energies.
 */
template<Ownership W, MemSpace M>
struct LivermorePEXsData
{
    template<class T>
    using Items = Collection<T, W, M>;
    template<class T>
    using ElementItems = Collection<T, W, M, ElementId>;

    //// MEMBER DATA ////

    Items<real_type>               reals;
    Items<LivermoreSubshell>       shells;
    ElementItems<LivermoreElement> elements;

    //// MEMBER FUNCTIONS ////

    //! Check whether the data is assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        return !reals.empty() && !shells.empty() && !elements.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    LivermorePEXsData& operator=(const LivermorePEXsData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        reals    = other.reals;
        shells   = other.shells;
        elements = other.elements;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Helper struct for making assignment easier
 */
struct LivermorePEIds
{
    //! Model ID
    ModelId model;
    //! ID of an electron
    ParticleId electron;
    //! ID of a gamma
    ParticleId gamma;

    //! Whether the IDs are assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        return model && electron && gamma;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Device data for creating a LivermorePEInteractor.
 */
template<Ownership W, MemSpace M>
struct LivermorePEData
{
    //// MEMBER DATA ////

    //! IDs in a separate struct for readability/easier copying
    LivermorePEIds ids;

    //! 1 / electron mass [1 / MevMass]
    real_type inv_electron_mass;

    //! Livermore EPICS2014 photoelectric data
    LivermorePEXsData<W, M> xs;

    //// MEMBER FUNCTIONS ////

    //! Check whether the data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return ids && inv_electron_mass > 0 && xs;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    LivermorePEData& operator=(const LivermorePEData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        ids               = other.ids;
        inv_electron_mass = other.inv_electron_mass;
        xs                = other.xs;
        return *this;
    }
};

using LivermorePEDeviceRef
    = LivermorePEData<Ownership::const_reference, MemSpace::device>;
using LivermorePEHostRef
    = LivermorePEData<Ownership::const_reference, MemSpace::host>;
using LivermorePEPointers
    = LivermorePEData<Ownership::const_reference, MemSpace::native>;

//---------------------------------------------------------------------------//
// KERNEL LAUNCHERS
//---------------------------------------------------------------------------//

// Launch the Livermore photoelectric interaction
void livermore_pe_interact(const LivermorePEDeviceRef&                pe,
                           const ModelInteractRefs<MemSpace::device>& model);

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
