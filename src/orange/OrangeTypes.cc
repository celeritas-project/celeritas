//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeTypes.cc
//---------------------------------------------------------------------------//
#include "OrangeTypes.hh"

#include <type_traits>

#include "corecel/Assert.hh"
#include "corecel/io/EnumStringMapper.hh"
#include "corecel/io/Logger.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/SoftEqual.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Whether the square of the tolerance rounds to zero for O(1) operations.
 *
 * This is an important criterion for construction and tracking operations that
 * involve \c sqrt, for example:
 * - Rotation matrix simplification
 * - Shape simplification
 * - Tracking to quadric surfaces
 */
template<class T>
constexpr bool is_tol_sq_nonnegligible(T value)
{
    return T{1} - ipow<2>(value) != T{1};
}

}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Use a relative error of \f$ \sqrt(\epsilon_\textrm{machine}) \f$ .
 *
 * Technically we're rounding the machine epsilon to a nearby value. We
 * could use numeric_limits<real_type>::epsilon instead.
 */
template<class T>
Tolerance<T> Tolerance<T>::from_default(real_type length)
{
    constexpr real_type sqrt_emach = [] {
        if constexpr (std::is_same_v<real_type, double>)
        {
            // std::sqrt(LimitsT::epsilon()) = 1.4901161193847656e-08
            return 1.5e-8;
        }
        else if constexpr (std::is_same_v<real_type, float>)
        {
            // std::sqrt(LimitsT::epsilon()) = 0.00034526698f
            return 3e-4f;
        }
    }();
    static_assert(is_tol_sq_nonnegligible(sqrt_emach),
                  "default tolerance is too low");

    return Tolerance<T>::from_relative(sqrt_emach, length);
}

//---------------------------------------------------------------------------//
/*!
 * Construct from the default "soft equivalence" tolerance.
 */
template<class T>
Tolerance<T> Tolerance<T>::from_softequal()
{
    constexpr SoftEqual<> default_seq{};
    return Tolerance<T>::from_relative(default_seq.rel(), default_seq.abs());
}

//---------------------------------------------------------------------------//
/*!
 * Construct from a relative tolerance and a length scale.
 */
template<class T>
Tolerance<T> Tolerance<T>::from_relative(real_type rel, real_type length)
{
    CELER_VALIDATE(rel > 0 && rel < 1,
                   << "tolerance " << rel
                   << " is out of range [must be in (0,1)]");
    CELER_VALIDATE(length > 0,
                   << "length scale " << length
                   << " is invalid [must be positive]");

    Tolerance<T> user;
    user.rel = rel;
    user.abs = rel * length;

    auto result = user.clamped();
    if (result.rel != user.rel)
    {
        CELER_LOG(warning) << "Clamped relative tolerance " << user.rel
                           << " to machine epsilon " << result.rel;
    }
    if (result.abs != user.abs)
    {
        CELER_LOG(warning) << "Clamping absolute tolerance " << user.abs
                           << " to minimum normal value " << result.abs;
    }

    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get a copy clamped to machine precision.
 *
 * Tolerances that are too tight may cause some deduplication logic to fail.
 * This checks and returns:
 * - relative error against machine epsilon, i.e., the relative
 *   difference between two adjacent floating point numbers, and
 * - absolute error against the floating point minimum, i.e., the smallest
 *   absolute magnitude that has a non-denormalized value.
 */
template<class T>
auto Tolerance<T>::clamped() const -> Tolerance<T>
{
    Tolerance<T> result;

    using LimitsT = std::numeric_limits<T>;
    result.rel = std::max(this->rel, LimitsT::epsilon());
    result.abs = std::max(this->abs, LimitsT::min());
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to a surface sense.
 */
char const* to_cstring(Sense s)
{
    if (s == Sense::inside)
        return "inside";
    else if (s == Sense::outside)
        return "outside";
    CELER_ASSERT_UNREACHABLE();
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
        "inv",
        // clang-format on
    };
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to a transform type.
 */
char const* to_cstring(TransformType value)
{
    static EnumStringMapper<TransformType> const to_cstring_impl{
        "no_transformation",
        "translation",
        "transformation",
    };
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to a logic notation
 */
char const* to_cstring(LogicNotation value)
{
    static EnumStringMapper<LogicNotation> const to_cstring_impl{
        "postfix",
        "infix",
    };
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to a transform type.
 */
char const* to_cstring(ZOrder zo)
{
    switch (zo)
    {
        case ZOrder::invalid:
            return "invalid";
        case ZOrder::background:
            return "background";
        case ZOrder::media:
            return "media";
        case ZOrder::array:
            return "array";
        case ZOrder::hole:
            return "hole";
        case ZOrder::implicit_exterior:
            return "implicit_exterior";
        case ZOrder::exterior:
            return "exterior";
    };
    CELER_VALIDATE(false, << "invalid ZOrder");
}

//---------------------------------------------------------------------------//
/*!
 * Get a printable character corresponding to a z ordering.
 */
char to_char(ZOrder zo)
{
    switch (zo)
    {
        case ZOrder::invalid:
            return '!';
        case ZOrder::background:
            return 'B';
        case ZOrder::media:
            return 'M';
        case ZOrder::array:
            return 'A';
        case ZOrder::hole:
            return 'H';
        case ZOrder::implicit_exterior:
            return 'x';
        case ZOrder::exterior:
            return 'X';
    };
    return '?';
}

//---------------------------------------------------------------------------//
/*!
 * Convert a printable character to a z ordering.
 */
ZOrder to_zorder(char c)
{
    // NOLINTNEXTLINE(bugprone-switch-missing-default-case)
    switch (c)
    {
        case '!':
            return ZOrder::invalid;
        case 'B':
            return ZOrder::background;
        case 'M':
            return ZOrder::media;
        case 'A':
            return ZOrder::array;
        case 'H':
            return ZOrder::hole;
        case 'x':
            return ZOrder::implicit_exterior;
        case 'X':
            return ZOrder::exterior;
    };
    return ZOrder::invalid;
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

template struct Tolerance<float>;
template struct Tolerance<double>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
