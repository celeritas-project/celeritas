//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/data/LivermorePEData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/Collection.hh"
#include "corecel/grid/NonuniformGridData.hh"
#include "corecel/math/Quantity.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"

namespace celeritas
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
    using XsUnits = units::Barn;
    using Energy = RealQuantity<EnergyUnits>;
    using Real6 = Array<real_type, 6>;

    // Binding energy of the electron
    Energy binding_energy;

    // Tabulated subshell photoionization cross section (used below 5 keV)
    NonuniformGridRecord xs;

    // Fit parameters for the integrated subshell photoionization cross
    // sections in the two different energy ranges (used above 5 keV)
    Array<Real6, 2> param;

    //! True if assigned and valid
    explicit CELER_FUNCTION operator bool() const
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
    NonuniformGridRecord xs_lo;

    // Total cross section above the K-shell energy but below the energy
    // threshold for the parameterized cross sections. Uses spline
    // interpolation.
    NonuniformGridRecord xs_hi;

    // SUBSHELL CROSS SECTIONS

    ItemRange<LivermoreSubshell> shells;

    // Energy threshold for using the parameterized subshell cross sections in
    // the lower and upper energy range
    Energy thresh_lo;  //!< Use tabulated XS below this energy
    Energy thresh_hi;  //!< Use lower parameterization below, upper above

    //! Energy below which cross sections are calculated on the fly
    static CELER_CONSTEXPR_FUNCTION Energy tabulated_threshold()
    {
        return Energy{0.2};
    }

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
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

    Items<LivermoreSubshell> shells;
    ElementItems<LivermoreElement> elements;

    // Backend data
    Items<real_type> reals;

    //// MEMBER FUNCTIONS ////

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return !shells.empty() && !elements.empty() && !reals.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    LivermorePEXsData& operator=(LivermorePEXsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        shells = other.shells;
        elements = other.elements;
        reals = other.reals;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Helper struct for making assignment easier
 */
struct LivermorePEIds
{
    //! ID of an electron
    ParticleId electron;
    //! ID of a gamma
    ParticleId gamma;

    //! Whether the IDs are assigned
    explicit CELER_FUNCTION operator bool() const { return electron && gamma; }
};

//---------------------------------------------------------------------------//
/*!
 * Device data for creating a LivermorePEInteractor.
 */
template<Ownership W, MemSpace M>
struct LivermorePEData
{
    using Mass = units::MevMass;

    //// MEMBER DATA ////

    //! IDs in a separate struct for readability/easier copying
    LivermorePEIds ids;

    //! 1 / electron mass [1 / Mass]
    real_type inv_electron_mass;

    //! Livermore EPICS2014 photoelectric data
    LivermorePEXsData<W, M> xs;

    //// MEMBER FUNCTIONS ////

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return ids && inv_electron_mass > 0 && xs;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    LivermorePEData& operator=(LivermorePEData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        ids = other.ids;
        inv_electron_mass = other.inv_electron_mass;
        xs = other.xs;
        return *this;
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
