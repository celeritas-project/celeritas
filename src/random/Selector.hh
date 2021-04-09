//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Selector.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>
#include "base/Range.hh"

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
 * accumulation algorithm. A "debug" version can check that the provided
 "total"
 * value isn't higher than the sum.
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
 */
template<class F, class T, bool D = CELERITAS_DEBUG>
class Selector
{
  public:
    //!@{
    //! Type aliases
    using value_type = T;
    using real_type  = typename std::result_of<F(value_type)>::type;
    //!@}

    constexpr static bool use_debug_sampling = D;

  public:
    // Construct with function, size, and accumulated value
    inline CELER_FUNCTION Selector(F&& eval, value_type size, real_type total);

    // Sample from the distribution
    template<class Engine>
    inline CELER_FUNCTION T operator()(Engine& rng) const;

  private:
    using IterT = RangeIter<T>;

    F         eval_;
    IterT     last_;
    real_type total_;
};

//---------------------------------------------------------------------------//
// Create a selector object from a function and total accumulated value
template<class F, class T>
inline CELER_FUNCTION Selector<F, T>
make_selector(F&& func, T size, decltype(func(size)) total = 1);

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "Selector.i.hh"
