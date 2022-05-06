//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/Quantity.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"

#include "NumericLimits.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
//! Implementation class for creating a nonnumeric value comparable to
//! Quantity.
template<class T>
struct UnitlessQuantity
{
    T value_; //!< Special nonnumeric value
};

//---------------------------------------------------------------------------//
} // namespace detail

//---------------------------------------------------------------------------//
/*!
 * A numerical value tagged with a unit.
 * \tparam UnitT  unit tag class
 * \tparam ValueT value type
 *
 * A quantity is a value expressed in terms of the given unit. Storing values
 * in a different unit system can help with some calculations (e.g. operating
 * in natural unit systems) by avoiding numerical multiplications and divisions
 * by large constants. It can also make debugging easier (numeric values are
 * obvious).
 *
 * Example usage by physics class, where charge is in units of q_e+, and
 * mass and momentum are expressed in atomic natural units (where m_e = 1 and c
 * = 1).
 * \code
    using MevEnergy        = Quantity<Mev>;
    using MevMass          = Quantity<UnitDivide<Mev, CLightSq>>;
    using MevMomentum      = Quantity<UnitDivide<Mev, CLight>>;
   \endcode
 *
 * A relativistic equation that operates on these quantities can do so without
 * unnecessary floating point operations involving the speed of light:
 * \code
   real_type eval = value_as<MevEnergy>(energy); // Natural units
   MevMomentum momentum{std::sqrt(eval * eval
                                  + 2 * value_as<MevMass>(mass) * eval)};
   \endcode
 * The resulting quantity can be converted to the native Celeritas unit system
 * with `native_value_from`, which multiplies in the constant value of
 * ElMomentumUnit:
 * \code
 * real_type mom = native_value_from(momentum);
 * \endcode
 *
 * When using a Quantity from another part of the code, e.g. an imported unit
 * system, use the \c quantity free function rather than \c .value() in order
 * to guarantee consistency of units between source and destination.
 *
 * \note The Quantity is designed to be a simple "strong type" class, not a
 * complex mathematical class. To operate on quantities, you must use
 `value_as`
 * (to operate within the Quantity's unit system) or `native_value_from` (to
 * operate in the Celeritas native unit system), use the resulting numeric
 * values in your mathematical expressions, then return a new Quantity class
 * with the resulting value and correct type.
 */
template<class UnitT, class ValueT = real_type>
class Quantity
{
  public:
    //!@{
    //! Type aliases
    using value_type = ValueT;
    using unit_type  = UnitT;
    using Unitless   = detail::UnitlessQuantity<ValueT>;
    //!@}

  public:
    //! Construct with default (zero)
    constexpr Quantity() = default;

    //! Construct with value in celeritas native units
    explicit CELER_CONSTEXPR_FUNCTION Quantity(value_type value)
        : value_(value)
    {
    }

    //! Construct implicitly from a unitless quantity
    CELER_CONSTEXPR_FUNCTION Quantity(Unitless uq) : value_(uq.value_) {}

    //! Get numeric value, discarding units
    CELER_CONSTEXPR_FUNCTION value_type value() const { return value_; }

  private:
    value_type value_{};
};

//---------------------------------------------------------------------------//
//! \cond
#define CELER_DEFINE_QUANTITY_CMP(TOKEN)                             \
    template<class U, class T>                                       \
    CELER_CONSTEXPR_FUNCTION bool operator TOKEN(Quantity<U, T> lhs, \
                                                 Quantity<U, T> rhs) \
    {                                                                \
        return lhs.value() TOKEN rhs.value();                        \
    }                                                                \
    template<class U, class T>                                       \
    CELER_CONSTEXPR_FUNCTION bool operator TOKEN(                    \
        Quantity<U, T> lhs, detail::UnitlessQuantity<T> rhs)         \
    {                                                                \
        return lhs.value() TOKEN rhs.value_;                         \
    }                                                                \
    template<class U, class T>                                       \
    CELER_CONSTEXPR_FUNCTION bool operator TOKEN(                    \
        detail::UnitlessQuantity<T> lhs, Quantity<U, T> rhs)         \
    {                                                                \
        return lhs.value_ TOKEN rhs.value();                         \
    }                                                                \
    namespace detail                                                 \
    {                                                                \
    template<class T>                                                \
    CELER_CONSTEXPR_FUNCTION bool                                    \
    operator TOKEN(UnitlessQuantity<T> lhs, UnitlessQuantity<T> rhs) \
    {                                                                \
        return lhs.value_ TOKEN rhs.value_;                          \
    }                                                                \
    }

//!@{
//! Comparisons for Quantity
CELER_DEFINE_QUANTITY_CMP(==)
CELER_DEFINE_QUANTITY_CMP(!=)
CELER_DEFINE_QUANTITY_CMP(<)
CELER_DEFINE_QUANTITY_CMP(>)
CELER_DEFINE_QUANTITY_CMP(<=)
CELER_DEFINE_QUANTITY_CMP(>=)
//!@}

#undef CELER_DEFINE_QUANTITY_CMP

//! \endcond
//---------------------------------------------------------------------------//
//! Value is C1::value() / C2::value()
template<class C1, class C2>
struct UnitDivide
{
    //! Get the conversion factor of the resulting unit
    static CELER_CONSTEXPR_FUNCTION auto value() -> decltype(auto)
    {
        return C1::value() / C2::value();
    }
};

//! Value is C1::value() * C2::value()
template<class C1, class C2>
struct UnitProduct
{
    //! Get the conversion factor of the resulting unit
    static CELER_CONSTEXPR_FUNCTION auto value() -> decltype(auto)
    {
        return C1::value() * C2::value();
    }
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Get a zero quantity (analogous to nullptr).
 */
CELER_CONSTEXPR_FUNCTION detail::UnitlessQuantity<real_type> zero_quantity()
{
    return {0};
}

//---------------------------------------------------------------------------//
/*!
 * Get a quantitity greater than any other numeric quantity.
 */
CELER_CONSTEXPR_FUNCTION detail::UnitlessQuantity<real_type> max_quantity()
{
    return {numeric_limits<real_type>::infinity()};
}

//---------------------------------------------------------------------------//
/*!
 * Get a quantitity less than any other numeric quantity.
 */
CELER_CONSTEXPR_FUNCTION detail::UnitlessQuantity<real_type> neg_max_quantity()
{
    return {-numeric_limits<real_type>::infinity()};
}

//---------------------------------------------------------------------------//
/*!
 * Swap two Quantities.
 */
template<class U, class V>
CELER_CONSTEXPR_FUNCTION void
swap(Quantity<U, V>& a, Quantity<U, V>& b) noexcept
{
    Quantity<U, V> tmp{a};
    a = b;
    b = tmp;
}

//---------------------------------------------------------------------------//
/*!
 * Convert the given quantity into the native Celeritas unit system.
 *
 * \code
 assert(native_value_from(Quantity<CLight>{1}) == 2.998e10 *
 centimeter/second);
 * \endcode
 */
template<class UnitT, class ValueT>
CELER_CONSTEXPR_FUNCTION auto native_value_from(Quantity<UnitT, ValueT> quant)
    -> decltype(auto)
{
    return quant.value() * UnitT::value();
}

//---------------------------------------------------------------------------//
/*!
 * Create a quantity from a value in the Celeritas unit system.
 *
 * This function can be used for defining a constant for use in another unit
 * system (typically a "natural" unit system for use in physics kernels).
 *
 * \code
 constexpr LightSpeed c = native_value_to<LightSpeed>(constants::c_light);
 assert(c.value() == 1);
 * \endcode
 */
template<class Q>
CELER_CONSTEXPR_FUNCTION Q native_value_to(typename Q::value_type value)
{
    using value_type = typename Q::value_type;
    using unit_type  = typename Q::unit_type;

    return Q{value * (value_type{1} / unit_type::value())};
}

//---------------------------------------------------------------------------//
/*!
 * Use the value of a Quantity.
 *
 * The redundant unit type in the function signature is to make coupling safer
 * across different parts of the code and to make the user code more readable.
 *
 * \code
 assert(value_as<LightSpeed>(LightSpeed{1}) == 1);
 * \endcode
 */
template<class Q, class SrcUnitT, class ValueT>
CELER_CONSTEXPR_FUNCTION auto value_as(Quantity<SrcUnitT, ValueT> quant)
    -> ValueT
{
    static_assert(std::is_same<Q, Quantity<SrcUnitT, ValueT>>::value,
                  "quantity units do not match");
    return quant.value();
}

//---------------------------------------------------------------------------//
} // namespace celeritas
