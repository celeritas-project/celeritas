//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MaterialInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Span.hh"
#include "base/Types.hh"
#include "physics/base/Units.hh"
#include "MaterialInterface.hh"
#include "Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// PARAMS
//---------------------------------------------------------------------------//
/*!
 * Fundamental, invariant properties of an element.
 *
 * Add elemental properties as needed if they apply to more than one physics
 * model. TODO:
 * - atomic shell
 * - isotopic components
 *
 * Note that more than one "element def" can exist for a single atomic number:
 * there might be different enrichments of an element in the problem.
 */
struct ElementDef
{
    int            atomic_number = 0; //!< Z number
    units::AmuMass atomic_mass;       //!< Isotope-weighted average atomic mass

    // COMPUTED PROPERTIES

    real_type cbrt_z   = 0; //!< Z^{1/3}
    real_type cbrt_zzp = 0; //!< (Z (Z + 1))^{1/3}
    real_type log_z    = 0; //!< log Z

    real_type coulomb_correction   = 0; //!< f(Z)
    real_type mass_radiation_coeff = 0; //!< 1/X_0 (bremsstrahlung)
};

//---------------------------------------------------------------------------//
/*!
 * Fractional element component of a material.
 *
 * This represents, e.g., the fraction of hydrogen in water.
 */
struct MatElementComponent
{
    ElementId element;  //!< Index in MaterialParams elements
    real_type fraction; //!< Fraction of number density
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
/*!
 * Access material properties on the device.
 *
 * This view is created from \c MaterialParams.
 *
 * \sa MaterialParams (owns the pointed-to data)
 * \sa ElementView (uses the pointed-to element data in a kernel)
 * \sa MaterialView (uses the pointed-to material data in a kernel)
 */
struct MaterialParamsPointers
{
    Span<const ElementDef>  elements;
    Span<const MaterialDef> materials;
    size_type               max_element_components;

    //! Check whether the interface is assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        return !materials.empty();
    }
};

//---------------------------------------------------------------------------//
// STATE
//---------------------------------------------------------------------------//
/*!
 * Dynamic material state of a particle track.
 */
struct MaterialTrackState
{
    MaterialId material_id; //!< Current material being tracked
};

//---------------------------------------------------------------------------//
/*!
 * View to the dynamic states of multiple physical particles.
 *
 * The size of the view will be the size of the vector of tracks. Each particle
 * track state corresponds to the thread ID (\c ThreadId).
 *
 * The "element scratch space" is a 2D array of reals, indexed with
 * [track_id][el_component_id], where the fast-moving dimension has the
 * greatest number of element components of any material in the problem. This
 * can be used for the physics to calculate microscopic cross sections.
 *
 * \sa MaterialStateStore (owns the pointed-to data)
 * \sa MaterialTrackView (uses the pointed-to data in a kernel)
 */
struct MaterialStatePointers
{
    Span<MaterialTrackState> state;
    Span<real_type> element_scratch; // 2D array: [num states][max components]

    //! Check whether the view is assigned
    explicit CELER_FUNCTION operator bool() const { return !state.empty(); }

    //! State size
    CELER_FUNCTION size_type size() const { return state.size(); }
};

//---------------------------------------------------------------------------//
} // namespace celeritas
