//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/distribution/detail/SelectorImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "corecel/cont/Range.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/SoftEqual.hh"
#include "corecel/random/distribution/GenerateCanonical.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
//! Whether to allow a "remainder" element.
enum class SelectorNormalization
{
    unnormalized,  //!< Components will not sum to total
    normalized  //!< Components should sum to total
};

template<SelectorNormalization N>
using SelNormTag = std::integral_constant<SelectorNormalization, N>;
using NormalizedSelectorTag = SelNormTag<SelectorNormalization::normalized>;
using UnnormalizedSelectorTag = SelNormTag<SelectorNormalization::unnormalized>;

//---------------------------------------------------------------------------//
/*!
 * Select a weighted discrete distribution on the fly.
 *
 * See \c make_selector .
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
    // Construct with function, size, accumulated value, and normalization
    template<SelectorNormalization N = SelectorNormalization::normalized>
    inline CELER_FUNCTION
    Selector(F&& eval, value_type size, real_type total, SelNormTag<N>);

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
 */
template<class F, class T>
template<SelectorNormalization N>
CELER_FUNCTION Selector<F, T>::Selector(F&& eval,
                                        value_type size,
                                        real_type total,
                                        SelNormTag<N>)
    : eval_{celeritas::forward<F>(eval)}, last_{size}, total_{total}
{
    CELER_EXPECT(last_ != IterT{});
    CELER_EXPECT(total_ > 0);
    if constexpr (CELERITAS_DEBUG)
    {
        real_type debug_total = 0;
        for (IterT iter{}; iter != last_; ++iter)
        {
            debug_total += eval_(*iter);
        }
        if constexpr (N == SelectorNormalization::normalized)
        {
            CELER_EXPECT(soft_equal(debug_total, total_));
        }
        else
        {
            CELER_EXPECT(debug_total <= total_
                         || soft_equal(debug_total, total_));
        }
    }

    if constexpr (N == SelectorNormalization::normalized)
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
}  // namespace detail
}  // namespace celeritas
