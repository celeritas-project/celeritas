//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MaterialDef.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/OpaqueId.hh"
#include "base/Span.hh"
#include "base/Types.hh"
#include "Units.hh"
#include "ElementDef.hh"

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
    ElementDefId element;  //!< Index in MaterialParams elements
    real_type    fraction; //!< Fraction of number density
};
using ElementComponentId = OpaqueId<MatElementComponent>;

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
    real_type density;        //!< Density [g/cm^3]
    real_type number_density; //!< Number density [at/cm^3]
    real_type temperature;    //!< Temperature [K]

    span<const MatElementComponent> elements; // Access by ElementComponentId
    // TODO for hadrons: span<const MatNuclideComponent> nuclides;

    // >>> DERIVATIVE PROPERTIES

    real_type electron_density;      // special units??
    real_type radiation_length_tsai; // [cm]
};

//! Opaque index to MaterialDef in a vector: represents a material ID
using MaterialDefId = OpaqueId<MaterialDef>;

//---------------------------------------------------------------------------//
} // namespace celeritas
