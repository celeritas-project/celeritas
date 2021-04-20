//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Selector.i.hh
//---------------------------------------------------------------------------//

#include "random/distributions/GenerateCanonical.hh"
#include "base/SoftEqual.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with function, size, and accumulated value.
 */
template<class F, class T>
CELER_FUNCTION
Selector<F, T>::Selector(F&& eval, value_type size, real_type total)
    : eval_{std::forward<F>(eval)}, last_{size}, total_{total}
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
CELER_FUNCTION auto Selector<F, T>::debug_accumulated_total() const
    -> real_type
{
    real_type accum = 0;
    for (IterT iter{}; iter != last_; ++iter)
    {
        accum += eval_(*iter);
    }
    return accum;
}

//---------------------------------------------------------------------------//
/*!
 * Create a selector object from a function and total accumulated value.
 */
template<class F, class T>
CELER_FUNCTION Selector<F, T>
               make_selector(F&& func, T size, decltype(func(size)) total)
{
    return {std::forward<F>(func), size, total};
}

//---------------------------------------------------------------------------//
} // namespace celeritas
