//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ElementSelector.i.hh
//---------------------------------------------------------------------------//

#include "base/Assert.hh"
#include "base/Range.hh"
#include "random/distributions/GenerateCanonical.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with material, xs calculator, and storage.
 */
template<class MicroXsCalc>
CELER_FUNCTION ElementSelector::ElementSelector(const MaterialView& material,
                                                MicroXsCalc&& calc_micro_xs,
                                                SpanReal      storage)
    : elements_(material.elements()), mat_xs_(0), el_xs_(storage.data())
{
    REQUIRE(!elements_.empty());
    REQUIRE(storage.size() >= material.num_elements());
    for (auto i : range<unsigned int>(elements_.size()))
    {
        const real_type micro_xs = calc_micro_xs(elements_[i].element);
        CHECK(micro_xs >= 0);
        const real_type frac = elements_[i].fraction;

        el_xs_[i] = micro_xs;
        mat_xs_ += micro_xs * frac;
    }
    ENSURE(mat_xs_ >= 0);
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
    real_type    accum_xs = -mat_xs_ * generate_canonical(rng);
    unsigned int i        = 0;
    unsigned int imax     = elements_.size() - 1;
    for (; i != imax; ++i)
    {
        accum_xs += elements_[i].fraction * el_xs_[i];
        if (accum_xs >= 0)
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
    return {el_xs_, elements_.size()};
}

//---------------------------------------------------------------------------//
} // namespace celeritas
