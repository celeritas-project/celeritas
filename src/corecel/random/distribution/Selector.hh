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
#include "corecel/random/distribution/GenerateCanonical.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Whether to allow a "remainder" element.
enum class SelectorNormalization
{
    unnormalized,  //!< Components will not sum to total
    normalized  //!< Components should sum to total
};

//---------------------------------------------------------------------------//
/*!
 * On-the-fly selection of a weighted discrete distribution.
 *
 * This algorithm encapsulates the loop for sampling from distributions
 * of a function <code>f(index) -> real</code> (usually with the
 * function prototype \c real_type(*)(size_type) ).
 * The index can be an integer, an enumeration, or an OpaqueId . The Selector
 * is constructed with the size of the distribution (but using the indexing
 * type).
 *
 * Edge cases are thoroughly tested (it will never iterate off the end if
 * normalized, even for incorrect values of the "total" probability/xs), and it
 * uses one fewer register than the typical accumulation algorithm.
 * When building with debug checking and normalization, the constructor asserts
 * that the provided "total" value is consistent.
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
 *
 * \sa celeritas::make_selector , celeritas::make_unnormalized_selector
 * \tparam F function type to sample from
 * \tparam T index type passed to the function
 */
template<class F, class T>
class Selector
{
  public:
    //!@{
    //! \name Type aliases
    using arg_type = T;
    using real_type = typename std::invoke_result<F, arg_type>::type;
    //!@}

  public:
    // Construct with function, size, accumulated value, and normalization
    inline CELER_FUNCTION
    Selector(F&& eval, arg_type size, real_type total, SelectorNormalization);

    // Sample from the distribution
    template<class Engine>
    inline CELER_FUNCTION T operator()(Engine& rng) const;

  private:
    using IterT = RangeIter<T>;

    F eval_;
    IterT last_;
    real_type total_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with function, size, and accumulated value.
 *
 * The \c SelectorNormalization argument determines whether to allow an "off
 * the end" value. If unnormalized, the sum of all evaluated arguments can be
 * nontrivially less than the given total.
 */
template<class F, class T>
CELER_FUNCTION Selector<F, T>::Selector(F&& eval,
                                        arg_type size,
                                        real_type total,
                                        SelectorNormalization norm)
    : eval_{celeritas::forward<F>(eval)}, last_{size}, total_{total}
{
    CELER_EXPECT(last_ != IterT{});
    CELER_EXPECT(total_ > 0);
    if constexpr (CELERITAS_DEBUG)
    {
        real_type debug_total = 0;
        for (IterT iter{}; iter != last_; ++iter)
        {
            real_type current = eval_(*iter);
            CELER_EXPECT(current >= 0);
            debug_total += current;
        }
        CELER_EXPECT((norm == SelectorNormalization::unnormalized
                      && debug_total <= total_)
                     || soft_equal(debug_total, total_));
    }

    if (norm == SelectorNormalization::normalized)
    {
        // Don't accumulate the last value: it is just there to assert that the
        // 'total' is not out-of-bounds
        --last_;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Sample from the distribution.
 */
template<class F, class T>
template<class Engine>
CELER_FUNCTION T Selector<F, T>::operator()(Engine& rng) const
{
    real_type accum = -total_ * generate_canonical<real_type>(rng);
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
 * Create a normalized on-the-fly discrete PDF sampler.
 */
template<class F, class T>
CELER_FUNCTION Selector<F, T>
make_selector(F&& func, T size, real_type total = 1)
{
    return {celeritas::forward<F>(func),
            size,
            total,
            SelectorNormalization::normalized};
}

//---------------------------------------------------------------------------//
/*!
 * Create an unnormalized selector that can return \c size if past the end.
 */
template<class F, class T>
CELER_FUNCTION Selector<F, T>
make_unnormalized_selector(F&& func, T size, real_type total)
{
    return {celeritas::forward<F>(func),
            size,
            total,
            SelectorNormalization::unnormalized};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
