//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ElementSelector.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Span.hh"
#include "base/Types.hh"
#include "MaterialView.hh"
#include "Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Make a weighted random selection of an element.
 *
 * The element chooser is for selecting an elemental component (atom) of a
 * material based on the microscopic cross section and the abundance fraction
 * of the element in the material.
 *
 * On construction, the element chooser uses the provided arguments to
 * precalculate all the microscopic cross sections in the given storage space.
 * The given function `calc_micro_xs` must accept a `ElementId` and return a
 * `real_type`, a non-negative microscopic cross section.
 *
 * The element chooser does \em not calculate macroscopic cross sections
 * because they're multiplied by fraction, not number density, and we only
 * care about the fractional abundances and cross section weighting.
 *
 * \code
    ElementSelector select_element(mat, calc_micro, storage);
    real_type total_macro_xs
        = select_element.material_micro_xs() *  mat.number_density();
    ElementComponentId id = select_element(rng);
    real_type selected_micro_xs =
 select_element.elemental_micro_xs()[el.get()]; ElementView el =
 mat.element_view(id);
    // use el.Z(), etc.
   \endcode
 *
 * Note that the units of the calculated microscopic cross section will be
 * identical to the units returned by `calc_micro_xs`. The macroscopic cross
 * section units (micro times \c mat.number_density() ) will be 1/cm if and
 * only if calc_micro units are cm^2.
 *
 * \todo Refactor to use Selector.
 */
class ElementSelector
{
  public:
    //!@{
    //! Type aliases
    using SpanReal      = Span<real_type>;
    using SpanConstReal = Span<const real_type>;
    //!@}

  public:
    // Construct with material, xs calculator, and storage.
    template<class MicroXsCalc>
    inline CELER_FUNCTION ElementSelector(const MaterialView& material,
                                          MicroXsCalc&&       calc_micro_xs,
                                          SpanReal micro_xs_storage);

    // Sample with the given RNG
    template<class Engine>
    inline CELER_FUNCTION ElementComponentId operator()(Engine& rng) const;

    //! Weighted material microscopic cross section
    CELER_FUNCTION real_type material_micro_xs() const { return material_xs_; }

    // Individual (unweighted) microscopic cross sections
    inline CELER_FUNCTION SpanConstReal elemental_micro_xs() const;

  private:
    Span<const MatElementComponent> elements_;
    real_type                       material_xs_;
    real_type*                      elemental_xs_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "ElementSelector.i.hh"
