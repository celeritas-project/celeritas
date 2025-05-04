//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/distribution/Selector.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "corecel/cont/Range.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/SoftEqual.hh"

#include "GenerateCanonical.hh"

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
 * The given function *must* return a consistent value for the same given
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
 */
template<class F, class T>
class Selector
{
  public:
    //!@{
    //! \name Type aliases
    using value_type = T;
    using real_type = typename std::invoke_result<F, value_type>::type;
    //!@}

  public:
    // Construct with function, size, and accumulated value
    inline CELER_FUNCTION Selector(F&& eval, value_type size, real_type total);

    // Sample from the distribution
    template<class Engine>
    inline CELER_FUNCTION T operator()(Engine& rng) const;

  private:
    using IterT = RangeIter<T>;

    F eval_;
    IterT last_;
    real_type total_;

    // Total value, for debug checking
    inline CELER_FUNCTION real_type debug_accumulated_total() const;
};

//---------------------------------------------------------------------------//
/*!
 * Create a selector object from a function and total accumulated value.
 */
template<class F, class T>
CELER_FUNCTION Selector<F, T>
make_selector(F&& func, T size, decltype(func(size)) total = 1)
{
    return {celeritas::forward<F>(func), size, total};
}

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with function, size, and accumulated value.
 */
template<class F, class T>
CELER_FUNCTION
Selector<F, T>::Selector(F&& eval, value_type size, real_type total)
    : eval_{celeritas::forward<F>(eval)}, last_{size}, total_{total}
{
    CELER_EXPECT(last_ != IterT{});
    CELER_EXPECT(total_ > 0);
    CELER_EXPECT(
        celeritas::soft_equal(this->debug_accumulated_total(), total_));

    // Don't accumulate the last value except to assert that the 'total'
    // isn't out-of-bounds
    --last_;
}

//---------------------------------------------------------------------------//
/*!
 * Sample from the distribution.
 */
template<class F, class T>
template<class Engine>
CELER_FUNCTION T Selector<F, T>::operator()(Engine& rng) const
{
    real_type accum = -total_ * generate_canonical(rng);
    for (IterT iter{}; iter != last_; ++iter)
    {
        accum += eval_(*iter);
        if (accum > 0)
            return *iter;
    }

    return *last_;
}

//---------------------------------------------------------------------------//
/*!
 * Accumulate total value for debug checking.
 *
 * This should *only* be used in the constructor before last_ is decremented.
 */
template<class F, class T>
CELER_FUNCTION auto Selector<F, T>::debug_accumulated_total() const -> real_type
{
    real_type accum = 0;
    for (IterT iter{}; iter != last_; ++iter)
    {
        accum += eval_(*iter);
    }
    return accum;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
