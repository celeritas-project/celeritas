//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/rasterize/detail/SafetyCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/math/NumericLimits.hh"

#include "../ImageData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Calculate the safety distance at an x,y position.
 *
 * The direction of the initialized state is "out of the page". The maximum
 * distance should generally be the length scale of the image.
 */
template<class GTV>
class SafetyCalculator
{
  public:
    //!@{
    //! \name Type aliases
    using ParamsRef = NativeCRef<ImageParamsData>;
    using StateRef = NativeRef<ImageStateData>;
    //!@}

  public:
    // Construct with geo track view
    inline CELER_FUNCTION
    SafetyCalculator(GTV&&, ParamsRef const&, real_type max_distance);

    // Calculate safety at an x, y index
    inline CELER_FUNCTION real_type operator()(size_type x, size_type y);

  private:
    GTV geo_;
    ImageParamsScalars const& scalars_;
    Real3 dir_;
    real_type max_distance_;
};

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//

template<class GTV>
CELER_FUNCTION
SafetyCalculator(GTV&&, NativeCRef<ImageParamsData> const&, real_type)
    -> SafetyCalculator<GTV>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with geo track view.
 */
template<class GTV>
CELER_FUNCTION SafetyCalculator<GTV>::SafetyCalculator(GTV&& geo,
                                                       ParamsRef const& params,
                                                       real_type max_distance)
    : geo_{celeritas::forward<GTV>(geo)}
    , scalars_{params.scalars}
    , dir_{make_unit_vector(cross_product(scalars_.down, scalars_.right))}
    , max_distance_{max_distance}
{
    CELER_ENSURE(max_distance_ > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate safety at an x, y coordinate.
 */
template<class GTV>
CELER_FUNCTION real_type SafetyCalculator<GTV>::operator()(size_type x,
                                                           size_type y)
{
    geo_ = [&] {
        auto calc_offset = [pw = scalars_.pixel_width](size_type i) {
            return pw * (static_cast<real_type>(i) + real_type(0.5));
        };

        GeoTrackInitializer init;
        init.pos = scalars_.origin;
        axpy(calc_offset(y), scalars_.down, &init.pos);
        axpy(calc_offset(x), scalars_.right, &init.pos);
        init.dir = dir_;
        return init;
    }();

    if (geo_.is_outside())
    {
        return numeric_limits<real_type>::quiet_NaN();
    }

    return geo_.find_safety(max_distance_);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
