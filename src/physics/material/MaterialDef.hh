//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MaterialDef.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Span.hh"
#include "base/Types.hh"
#include "physics/base/Units.hh"
#include "ElementDef.hh"
#include "Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Fractional element component of a material.
 *
 * This represents, e.g., the fraction of hydrogen in water.
 */
struct MatElementComponent
{
    ElementId    element;  //!< Index in MaterialParams elements
    real_type    fraction; //!< Fraction of number density
};

//---------------------------------------------------------------------------//
/*!
 * Fundamental (static) properties of a material.
 *
 * Multiple material definitions are allowed to reuse a single element
 * definition vector (memory management from the params store should handle
 * this). Derivative properties such as electron_density are calculated from
 * the elemental components.
 */
struct MaterialDef
{
    real_type   number_density; //!< Atomic number density [1/cm^3]
    real_type   temperature;    //!< Temperature [K]
    MatterState matter_state;   //!< Solid, liquid, gas
    Span<const MatElementComponent> elements; //!< Access by ElementComponentId

    // COMPUTED PROPERTIES

    real_type density;          //!< Density [g/cm^3]
    real_type electron_density; //!< Electron number density [1/cm^3]
    real_type rad_length;       //!< Radiation length [cm]
};

//---------------------------------------------------------------------------//
} // namespace celeritas
