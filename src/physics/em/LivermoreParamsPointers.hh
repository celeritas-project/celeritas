//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LivermoreParamsPointers.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Span.hh"
#include "base/Types.hh"
#include "physics/base/Units.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Electron subshell data
 */
struct LivermoreSubshell
{
    // Binding energy of the electron
    units::MevEnergy binding_energy;

    // Tabulated subshell photoionization cross section (used below 5 keV)
    // TODO: value grid
    span<const real_type> energy;
    span<const real_type> xs;

    // Fit parameters for the integrated subshell photoionization cross
    // sections in the two different energy ranges (used above 5 keV)
    span<const real_type> param_low;
    span<const real_type> param_high;
};

//---------------------------------------------------------------------------//
/*!
 * Elemental photoelectric cross sections for the Livermore model.
 */
struct LivermoreElement
{
    // TOTAL CROSS SECTIONS

    // Total cross section below the K-shell energy. Uses linear interpolation.
    span<const real_type> energy_low;
    span<const real_type> xs_low;

    // Total cross section above the K-shell energy but below the energy
    // threshold for the parameterized cross sections. Uses spline
    // interpolation.
    span<const real_type> energy_high;
    span<const real_type> xs_high;

    // SUBSHELL CROSS SECTIONS

    span<const LivermoreSubshell> shells;

    // Energy threshold for using the parameterized subshell cross sections in
    // the lower and upper energy range
    units::MevEnergy thresh_low;
    units::MevEnergy thresh_high;
};

//---------------------------------------------------------------------------//
/*!
 * Access Livermore data on device.
 */
struct LivermoreParamsPointers
{
    span<const LivermoreElement> elements;

    //! Check whether the interface is assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        return !elements.empty();
    }
};

//---------------------------------------------------------------------------//
} // namespace celeritas
