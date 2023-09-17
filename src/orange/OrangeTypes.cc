//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeTypes.cc
//---------------------------------------------------------------------------//
#include "OrangeTypes.hh"

#include <type_traits>

#include "corecel/Assert.hh"
#include "corecel/io/EnumStringMapper.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/SoftEqual.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Use a relative error of \f$ \sqrt(\epsilon_\textrm{machine}) \f$ .
 *
 * Technically we're rounding the machine epsilon to a nearby power of 10. We
 * could use numeric_limits<real_type>::epsilon instead.
 */
template<class T>
Tolerances<T> Tolerances<T>::from_default(real_type length)
{
    constexpr real_type sqrt_emach = [] {
        if constexpr (std::is_same_v<real_type, double>)
        {
            return 1.e-8;
        }
        else if constexpr (std::is_same_v<real_type, float>)
        {
            return 5.e-3f;
        }
    }();
    static_assert(real_type{1} - ipow<2>(sqrt_emach) != real_type{1},
                  "default tolerance is insufficient");

    return Tolerances<T>::from_relative(sqrt_emach, length);
}

//---------------------------------------------------------------------------//
/*!
 * Construct from the default "soft equivalence" tolerance.
 */
template<class T>
Tolerances<T> Tolerances<T>::from_softequal()
{
    constexpr SoftEqual<> default_seq{};
    return Tolerances<T>::from_relative(default_seq.rel(), default_seq.abs());
}

//---------------------------------------------------------------------------//
/*!
 * Construct from a relative tolerance and a length scale.
 */
template<class T>
Tolerances<T> Tolerances<T>::from_relative(real_type rel, real_type length)
{
    CELER_VALIDATE(rel > 0 && rel < 1,
                   << "tolerance " << rel
                   << " is out of range [must be in (0,1)]");
    CELER_VALIDATE(length > 0,
                   << "length scale " << length
                   << " is invalid [must be positive]");
    Tolerances<T> result;
    result.rel = rel;
    result.abs = rel * length;
    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to a surface type.
 */
char const* to_cstring(SurfaceType value)
{
    static EnumStringMapper<SurfaceType> const to_cstring_impl{
        // clang-format off
        "px",
        "py",
        "pz",
        "cxc",
        "cyc",
        "czc",
        "sc",
        "cx",
        "cy",
        "cz",
        "p",
        "s",
        "kx",
        "ky",
        "kz",
        "sq",
        "gq",
        // clang-format on
    };
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

template struct Tolerances<float>;
template struct Tolerances<double>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
