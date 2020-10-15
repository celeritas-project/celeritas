//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PhotoelectricInteractorPointers.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Types.hh"

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
    span<const real_type> xs;
    span<const real_type> energy;

    // Fit parameters for the integrated subshell photoionization cross
    // sections in the two different energy ranges (used above 5 keV)
    span<const real_type> param_low;
    span<const real_type> param_high;
};

//---------------------------------------------------------------------------//
/*!
 * Elemental photoelectric cross sections for the Livermore model.
 */
struct LivermoreData
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
    // the lower and upper nergy range
    units::MevEnergy thresh_low;
    units::MevEnergy thresh_high;
};

//---------------------------------------------------------------------------//
/*!
 * Device data for creating a PhotoelectricInteractor.
 */
struct PhotoelectricInteractorPointers
{
    //! 1 / electron mass [1 / MevMass]
    real_type inv_electron_mass;
    //! ID of an electron
    ParticleDefId electron_id;
    //! ID of a gamma
    ParticleDefId gamma_id;
    //! Photoelectric cross section data. Size is the number of elements.
    span<const LivermoreData> elements;

    //! Check whether the data is assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        return electron_id && gamma_id && !elements.empty();
    }
};

//---------------------------------------------------------------------------//
} // namespace celeritas
