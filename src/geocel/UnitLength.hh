//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/UnitLength.hh
//---------------------------------------------------------------------------//
#pragma once

#include <ostream>

#include "corecel/math/Constant.hh"

#include "detail/LengthUnits.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Unit system used for reference output or test results.
 */
struct UnitLength
{
    Constant value{lengthunits::centimeter};
    char const* label{"cm"};

    template<class T>
    constexpr T from_native(T const& v) const
    {
        return v / value;
    }
};

//! Native unit length
inline constexpr UnitLength native_unit_length{Constant{1},
                                               lengthunits::native_label};

//---------------------------------------------------------------------------//
/*!
 * Print a length/position as a quantity with units.
 */
template<class T>
struct StreamableLength
{
    T const& native_value;
    UnitLength const& units;

    friend std::ostream& operator<<(std::ostream& os,
                                    StreamableLength const& sl)
    {
        os << sl.units.from_native(sl.native_value) << " [" << sl.units.label
           << ']';
        return os;
    }
};

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES (C++17)
//---------------------------------------------------------------------------//

template<class T>
StreamableLength(T const&, UnitLength) -> StreamableLength<T>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
