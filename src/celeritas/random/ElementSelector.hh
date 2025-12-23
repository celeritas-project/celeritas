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
#include "corecel/random/distribution/Selector.hh"
#include "celeritas/Types.hh"
#include "celeritas/mat/MaterialView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Select an element from a material using weighted cross sections.
 *
 * The element selector chooses a component (atomic element) of a
 * material based on the microscopic cross section and the abundance fraction
 * of the element in the material.
 *
 * On construction, the element chooser uses the provided arguments to
 * precalculate all the microscopic cross sections in the given storage space.
 * The given function \c calc_micro_xs must accept an \c ElementId and return a
 * microscopic cross section in \c units::BarnXs .
 *
 * Typical usage:
 * \code
    ElementSelector select_element(mat, calc_micro, mat.element_scratch());
    ElementComponentId id = select_element(rng);
    ElementView el = mat.element_record(id);
   \endcode
 */
class ElementSelector
{
  public:
    //!@{
    //! \name Type aliases
    using SpanReal = Span<real_type>;
    using MicroXs = units::BarnXs;
    //!@}

  public:
    // Construct with material, xs calculator, and storage.
    template<class MicroXsCalc>
    inline CELER_FUNCTION ElementSelector(MaterialView const& material,
                                          MicroXsCalc&& calc_micro_xs,
                                          SpanReal micro_xs_storage);

    // Sample with the given RNG
    template<class Engine>
    CELER_FORCEINLINE_FUNCTION ElementComponentId operator()(Engine& rng) const
    {
        return select_component_(rng);
    }

  private:
    //// TYPES ////

    struct MicroXsComponentGetter
    {
        Span<MatElementComponent const> elements_;
        real_type const* elemental_xs_{nullptr};

        inline CELER_FUNCTION real_type operator()(ElementComponentId) const;
    };
    using SelectorT = Selector<MicroXsComponentGetter, ElementComponentId>;

    //// DATA ////

    SelectorT select_component_;

    //// HELPER FUNCTIONS ////

    // Fill storage with micro xs, and return the accumulated weighted xs
    template<class MicroXsCalc>
    static inline CELER_FUNCTION MicroXs
    store_and_calc_xs(Span<MatElementComponent const> elements,
                      MicroXsCalc&& calc_micro_xs,
                      SpanReal storage);
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
    : select_component_{{material.elements(), storage.data()},
                        id_cast<ElementComponentId>(material.num_elements()),
                        value_as<MicroXs>(ElementSelector::store_and_calc_xs(
                            material.elements(),
                            celeritas::forward<MicroXsCalc>(calc_micro_xs),
                            storage)),
                        SelectorNormalization::normalized}
{
    CELER_EXPECT(material.num_elements() > 0);
}
//---------------------------------------------------------------------------//
/*!
 * Fill storage with micro xs, and return the weighted micro xs.
 *
 * This is called by the constructor.
 */
template<class MicroXsCalc>
CELER_FUNCTION auto
ElementSelector::store_and_calc_xs(Span<MatElementComponent const> elements,
                                   MicroXsCalc&& calc_micro_xs,
                                   SpanReal storage) -> MicroXs
{
    CELER_EXPECT(storage.size() >= elements.size());
    real_type total_xs{0};
    for (auto i : range<size_type>(elements.size()))
    {
        auto micro_xs = value_as<MicroXs>(calc_micro_xs(elements[i].element));
        CELER_ASSERT(micro_xs >= 0);
        storage[i] = micro_xs;
        total_xs += micro_xs * elements[i].fraction;
    }
    return MicroXs{total_xs};
}

//---------------------------------------------------------------------------//
/*!
 * Get weighted cross section for the given element component.
 */
CELER_FUNCTION real_type
ElementSelector::MicroXsComponentGetter::operator()(ElementComponentId i) const
{
    CELER_EXPECT(i < elements_.size());
    return elements_[i.unchecked_get()].fraction
           * elemental_xs_[i.unchecked_get()];
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
