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
template<class F, class T, bool D>
CELER_FUNCTION
Selector<F, T, D>::Selector(F&& eval, value_type size, real_type total)
    : eval_{std::forward<F>(eval)}, last_{size}, total_{total}
{
    CELER_EXPECT(last_ != IterT{});
    CELER_EXPECT(total_ >= 0);

    // Don't accumulate the last value except to assert that the 'total'
    // isn't out-of-bounds
    --last_;
}

//---------------------------------------------------------------------------//
/*!
 * Sample from the distribution.
 */
template<class F, class T, bool D>
template<class Engine>
CELER_FUNCTION T Selector<F, T, D>::operator()(Engine& rng) const
{
    if (!use_debug_sampling)
    {
        real_type accum = -total_ * generate_canonical(rng);
        for (IterT iter{}; iter != last_; ++iter)
        {
            accum += eval_(*iter);
            if (accum > 0)
                return *iter;
        }
    }
    else
    {
        // Equivalent to opt, but uses one more register and checks final value
        real_type accum = 0;
        real_type stop  = total_ * generate_canonical(rng);
        for (IterT iter{}; iter != last_; ++iter)
        {
            accum += eval_(*iter);
            if (accum > stop)
                return *iter;
        }

        // Check that the final value works and sums up to the expected total
        accum += eval_(*last_);
        CELER_ENSURE(celeritas::soft_equal(accum, total_));
    }
    return *last_;
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
