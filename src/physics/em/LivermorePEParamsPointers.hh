//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LivermorePEParamsPointers.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Span.hh"
#include "base/Types.hh"
#include "physics/base/Units.hh"
#include "MockXsCalculator.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Electron subshell data.
 */
struct LivermoreSubshell
{
    // Binding energy of the electron
    units::MevEnergy binding_energy;

    // Tabulated subshell photoionization cross section (used below 5 keV)
    // TODO: value grid
    ValueGrid xs;

    // Fit parameters for the integrated subshell photoionization cross
    // sections in the two different energy ranges (used above 5 keV)
    Span<const real_type> param_low;
    Span<const real_type> param_high;
};

//---------------------------------------------------------------------------//
/*!
 * Elemental photoelectric cross sections for the Livermore model.
 */
struct LivermoreElement
{
    // TOTAL CROSS SECTIONS

    // Total cross section below the K-shell energy. Uses linear interpolation.
    // TODO: value grid
    ValueGrid xs_low;

    // Total cross section above the K-shell energy but below the energy
    // threshold for the parameterized cross sections. Uses spline
    // interpolation.
    // TODO: value grid
    ValueGrid xs_high;

    // SUBSHELL CROSS SECTIONS

    Span<const LivermoreSubshell> shells;

    // Energy threshold for using the parameterized subshell cross sections in
    // the lower and upper energy range
    units::MevEnergy thresh_low;
    units::MevEnergy thresh_high;
};

//---------------------------------------------------------------------------//
/*!
 * Access Livermore data on device.
 */
struct LivermorePEParamsPointers
{
    Span<const LivermoreElement> elements;

    //! Check whether the interface is assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        return !elements.empty();
    }
};

//---------------------------------------------------------------------------//
} // namespace celeritas
