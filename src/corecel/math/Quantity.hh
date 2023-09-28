//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
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
    T value_;  //!< Special nonnumeric value
};

//! Helper class for getting attributes about a member function
template<class T>
struct AccessorTraits;

//! \cond
//! Access the return type using AccessorTraits<decltype(&Foo::bar)>
template<class ResultType, class ClassType>
struct AccessorTraits<ResultType (ClassType::*)() const>
{
    using type = ClassType;
    using result_type = ResultType;
};
//! \endcond

//! Get the result type of a class accessor
template<class T>
using AccessorResultType = typename AccessorTraits<T>::result_type;

//---------------------------------------------------------------------------//
}  // namespace detail

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
   using MevEnergy   = Quantity<Mev>;
   using MevMass     = Quantity<UnitDivide<Mev, CLightSq>>;
   using MevMomentum = Quantity<UnitDivide<Mev, CLight>>;
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
   real_type mom = native_value_from(momentum);
 * \endcode
 *
 * When using a Quantity from another part of the code, e.g. an imported unit
 * system, use the \c quantity free function rather than \c .value() in order
 * to guarantee consistency of units between source and destination.
 *
 * An example unit class would be:
 * \code
    struct DozenUnit
    {
        static constexpr int value() { return 12; }
        static constexpr char const* label() { return "dozen"; }
    };
   \endcode
 *
 * The label is used solely for outputting to JSON.
 *
 * \note The Quantity is designed to be a simple "strong type" class, not a
 * complex mathematical class. To operate on quantities, you must use
 * `value_as`
 * (to operate within the Quantity's unit system) or `native_value_from` (to
 * operate in the Celeritas native unit system), use the resulting numeric
 * values in your mathematical expressions, then return a new Quantity class
 * with the resulting value and correct type.
 */
template<class UnitT, class ValueT = decltype(UnitT::value())>
class Quantity
{
  public:
    //!@{
    //! \name Type aliases
    using value_type = ValueT;
    using unit_type = UnitT;
    using Unitless = detail::UnitlessQuantity<ValueT>;
    //!@}

  public:
    //! Construct with default (zero)
    constexpr Quantity() = default;

    //! Construct with value in celeritas native units
    explicit CELER_CONSTEXPR_FUNCTION Quantity(value_type value) noexcept
        : value_(value)
    {
    }

    //! Construct implicitly from a unitless quantity
    CELER_CONSTEXPR_FUNCTION Quantity(Unitless uq) noexcept : value_(uq.value_)
    {
    }

    //!@{
    //! Access the underlying numeric value, discarding units
#define CELER_DEFINE_QACCESS(FUNC, QUAL)                          \
    CELER_CONSTEXPR_FUNCTION value_type QUAL FUNC() QUAL noexcept \
    {                                                             \
        return value_;                                            \
    }

    CELER_DEFINE_QACCESS(value, &)
    CELER_DEFINE_QACCESS(value, const&)
#undef CELER_DEFINE_QACCESS
    //!@}

  private:
    value_type value_{};
};

//---------------------------------------------------------------------------//
//! \cond
#define CELER_DEFINE_QUANTITY_CMP(TOKEN)                                      \
    template<class U, class T, class T2>                                      \
    CELER_CONSTEXPR_FUNCTION bool operator TOKEN(                             \
        Quantity<U, T> lhs, Quantity<U, T2> rhs) noexcept                     \
    {                                                                         \
        return lhs.value() TOKEN rhs.value();                                 \
    }                                                                         \
    template<class U, class T>                                                \
    CELER_CONSTEXPR_FUNCTION bool operator TOKEN(                             \
        Quantity<U, T> lhs, detail::UnitlessQuantity<T> rhs) noexcept         \
    {                                                                         \
        return lhs.value() TOKEN rhs.value_;                                  \
    }                                                                         \
    template<class U, class T>                                                \
    CELER_CONSTEXPR_FUNCTION bool operator TOKEN(                             \
        detail::UnitlessQuantity<T> lhs, Quantity<U, T> rhs) noexcept         \
    {                                                                         \
        return lhs.value_ TOKEN rhs.value();                                  \
    }                                                                         \
    namespace detail                                                          \
    {                                                                         \
    template<class T>                                                         \
    CELER_CONSTEXPR_FUNCTION bool                                             \
    operator TOKEN(UnitlessQuantity<T> lhs, UnitlessQuantity<T> rhs) noexcept \
    {                                                                         \
        return lhs.value_ TOKEN rhs.value_;                                   \
    }                                                                         \
    }

//!@{
//! Comparison for Quantity
CELER_DEFINE_QUANTITY_CMP(==)
CELER_DEFINE_QUANTITY_CMP(!=)
CELER_DEFINE_QUANTITY_CMP(<)
CELER_DEFINE_QUANTITY_CMP(>)
CELER_DEFINE_QUANTITY_CMP(<=)
CELER_DEFINE_QUANTITY_CMP(>=)
//!@}

#undef CELER_DEFINE_QUANTITY_CMP

//!@{
//! Math operator for Quantity
template<class U, class T, class T2>
CELER_CONSTEXPR_FUNCTION auto
operator+(Quantity<U, T> lhs, Quantity<U, T2> rhs) noexcept -> decltype(auto)
{
    return Quantity<U, std::common_type_t<T, T2>>{lhs.value() + rhs.value()};
}

template<class U, class T, class T2>
CELER_CONSTEXPR_FUNCTION auto
operator-(Quantity<U, T> lhs, Quantity<U, T2> rhs) noexcept -> decltype(auto)
{
    return Quantity<U, std::common_type_t<T, T2>>{lhs.value() - rhs.value()};
}

template<class U, class T>
CELER_CONSTEXPR_FUNCTION auto operator-(Quantity<U, T> q) noexcept
    -> Quantity<U, T>
{
    return Quantity<U, T>{-q.value()};
}

template<class U, class T, class T2>
CELER_CONSTEXPR_FUNCTION auto operator*(Quantity<U, T> lhs, T2 rhs) noexcept
    -> decltype(auto)
{
    return Quantity<U, std::common_type_t<T, T2>>{lhs.value() * rhs};
}

template<class U, class T, class T2>
CELER_CONSTEXPR_FUNCTION auto operator*(T rhs, Quantity<U, T> lhs) noexcept
    -> decltype(auto)
{
    return Quantity<U, std::common_type_t<T, T2>>{rhs * lhs.value()};
}

template<class U, class T, class T2>
CELER_CONSTEXPR_FUNCTION auto operator/(Quantity<U, T> lhs, T2 rhs) noexcept
    -> decltype(auto)
{
    return Quantity<U, std::common_type_t<T, T2>>{lhs.value() / rhs};
}
//!@!}

//! \endcond
//---------------------------------------------------------------------------//
//! Value is C1::value() / C2::value()
template<class C1, class C2>
struct UnitDivide
{
    //! Get the conversion factor of the resulting unit
    static CELER_CONSTEXPR_FUNCTION auto value() noexcept -> decltype(auto)
    {
        return C1::value() / C2::value();
    }
};

//! Value is C1::value() * C2::value()
template<class C1, class C2>
struct UnitProduct
{
    //! Get the conversion factor of the resulting unit
    static CELER_CONSTEXPR_FUNCTION auto value() noexcept -> decltype(auto)
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
CELER_CONSTEXPR_FUNCTION detail::UnitlessQuantity<real_type>
zero_quantity() noexcept
{
    return {0};
}

//---------------------------------------------------------------------------//
/*!
 * Get a quantitity greater than any other numeric quantity.
 */
CELER_CONSTEXPR_FUNCTION detail::UnitlessQuantity<real_type>
max_quantity() noexcept
{
    return {numeric_limits<real_type>::infinity()};
}

//---------------------------------------------------------------------------//
/*!
 * Get a quantitity less than any other numeric quantity.
 */
CELER_CONSTEXPR_FUNCTION detail::UnitlessQuantity<real_type>
neg_max_quantity() noexcept
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
CELER_CONSTEXPR_FUNCTION auto
native_value_from(Quantity<UnitT, ValueT> quant) noexcept -> decltype(auto)
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
CELER_CONSTEXPR_FUNCTION Q native_value_to(typename Q::value_type value) noexcept
{
    return Q{value / Q::unit_type::value()};
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
CELER_CONSTEXPR_FUNCTION auto
value_as(Quantity<SrcUnitT, ValueT> quant) noexcept -> ValueT
{
    static_assert(std::is_same<Q, Quantity<SrcUnitT, ValueT>>::value,
                  "quantity units do not match");
    return quant.value();
}

//---------------------------------------------------------------------------//
/*!
 * Get the label for a unit returned from a class accessor.
 *
 * Example:
 * \code
   cout << accessor_unit_label<&ParticleView::mass>() << endl;
   \endcode
 */
template<class T>
inline char const* accessor_unit_label()
{
    return detail::AccessorResultType<T>::unit_type::label();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
