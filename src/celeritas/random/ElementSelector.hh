//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/random/ElementSelector.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/cont/Span.hh"
#include "corecel/random/distribution/GenerateCanonical.hh"
#include "celeritas/Types.hh"
#include "celeritas/mat/MaterialView.hh"

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
        = select_element.material_micro_xs() * mat.number_density();
    ElementComponentId id = select_element(rng);
    real_type selected_micro_xs
        = select_element.elemental_micro_xs()[el.get()];
    ElementView el = mat.element_record(id);
    // use el.Z(), etc.
   \endcode
 *
 * Note that the units of the calculated microscopic cross section will be
 * identical to the units returned by `calc_micro_xs`. The macroscopic cross
 * section units (micro times \c mat.number_density() ) will be 1/len if and
 * only if calc_micro units are len^2.
 *
 * \todo Refactor to use Selector.
 */
class ElementSelector
{
  public:
    //!@{
    //! \name Type aliases
    using SpanReal = Span<real_type>;
    using SpanConstReal = LdgSpan<real_type const>;
    //!@}

  public:
    // Construct with material, xs calculator, and storage.
    template<class MicroXsCalc>
    inline CELER_FUNCTION ElementSelector(MaterialView const& material,
                                          MicroXsCalc&& calc_micro_xs,
                                          SpanReal micro_xs_storage);

    // Sample with the given RNG
    template<class Engine>
    inline CELER_FUNCTION ElementComponentId operator()(Engine& rng) const;

    //! Weighted material microscopic cross section
    CELER_FUNCTION real_type material_micro_xs() const { return material_xs_; }

    // Individual (unweighted) microscopic cross sections
    inline CELER_FUNCTION SpanConstReal elemental_micro_xs() const;

  private:
    Span<MatElementComponent const> elements_;
    real_type material_xs_{0};
    real_type* elemental_xs_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with material, xs calculator, and storage.
 */
template<class MicroXsCalc>
CELER_FUNCTION ElementSelector::ElementSelector(MaterialView const& material,
                                                MicroXsCalc&& calc_micro_xs,
                                                SpanReal storage)
    : elements_(material.elements()), elemental_xs_(storage.data())
{
    CELER_EXPECT(!elements_.empty());
    CELER_EXPECT(storage.size() >= material.num_elements());
    for (auto i : range<size_type>(elements_.size()))
    {
        real_type const micro_xs
            = (calc_micro_xs(elements_[i].element)).value();
        CELER_ASSERT(micro_xs >= 0);
        real_type const frac = elements_[i].fraction;

        elemental_xs_[i] = micro_xs;
        material_xs_ += micro_xs * frac;
    }
    CELER_ENSURE(material_xs_ >= 0);
}

//---------------------------------------------------------------------------//
/*!
 * Sample the element with the given RNG.
 *
 * To reduce register usage, this function starts with the cumulative sums and
 * counts backward.
 */
template<class Engine>
CELER_FUNCTION ElementComponentId ElementSelector::operator()(Engine& rng) const
{
    real_type accum_xs = -material_xs_ * generate_canonical(rng);
    size_type i = 0;
    size_type imax = elements_.size() - 1;
    for (; i != imax; ++i)
    {
        accum_xs += elements_[i].fraction * elemental_xs_[i];
        if (accum_xs > 0)
            break;
    }
    return ElementComponentId{i};
}

//---------------------------------------------------------------------------//
/*!
 * Get individual microscopic cross sections.
 */
CELER_FUNCTION auto ElementSelector::elemental_micro_xs() const -> SpanConstReal
{
    return {elemental_xs_, elements_.size()};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
