//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/distribution/Selector.hh
//---------------------------------------------------------------------------//
#pragma once

#include "detail/SelectorImpl.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * On-the-fly selection of a weighted discrete distribution.
 *
 * This algorithm encapsulates the loop for sampling from distributions by
 * integer index or by OpaqueId. Edge cases are thoroughly tested (it will
 * never iterate off the end, even for incorrect values of the "total"
 * probability/xs), and it uses one fewer register than the typical
 * accumulation algorithm. When building with debug checking, the constructor
 * asserts that the provided "total" value is consistent.
 *
 * The given function \em must return a consistent value for the same given
 * argument.
 *
 * \code
    auto select_el = make_selector(
        [](ElementId i) { return xs[i.get()]; },
        ElementId{num_elements()},
        tot_xs);
    ElementId el = select_el(rng);
   \endcode
 * or
 * \code
    auto select_val = make_selector([](size_type i) { return pdf[i]; },
                                    pdf.size());
    size_type idx = select_val(rng);
   \endcode
 * Create a normalized selector from a function and total accumulated value.
 */
template<class F, class T>
CELER_FUNCTION detail::Selector<F, T>
make_selector(F&& func, T size, real_type total = 1)
{
    using NormTag = detail::NormalizedSelectorTag;
    return {celeritas::forward<F>(func), size, total, NormTag{}};
}

//---------------------------------------------------------------------------//
/*!
 * Create an unnormalized selector that can return \c size if past the end.
 */
template<class F, class T>
CELER_FUNCTION detail::Selector<F, T>
make_unnormalized_selector(F&& func, T size, real_type total)
{
    using NormTag = detail::UnnormalizedSelectorTag;
    return {celeritas::forward<F>(func), size, total, NormTag{}};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
