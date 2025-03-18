//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/Quantity.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"

#include "detail/QuantityImpl.hh"

namespace celeritas
{
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
 * Example usage by physics class, where charge is in units of \f$ q_{e^+} \f$,
 * and mass and momentum are expressed in atomic natural units (where
 * \f$ m_e = 1 \f$ and \f$ c = 1 \f$).
 * \code
   using MevEnergy   = Quantity<Mev, real_type>;
   using MevMass     = RealQuantity<UnitDivide<Mev, CLightSq>>;
   using MevMomentum = RealQuantity<UnitDivide<Mev, CLight>>;
   \endcode
 *
 * Note the use of the \c RealQuantity type alias (below).
 *
 * A relativistic equation that operates on these quantities can do so without
 * unnecessary floating point operations involving the speed of light:
 * \code
   real_type e_mev = value_as<MevEnergy>(energy); // Natural units
   MevMomentum momentum{std::sqrt(e_mev * e_mev
                                  + 2 * value_as<MevMass>(mass) * e_mev)};
   \endcode
 * The resulting quantity can be converted to the native Celeritas unit system
 * with `native_value_from`, which multiplies in the constant value of
 * ElMomentumUnit:
 * \code
   real_type mom = native_value_from(momentum);
 * \endcode
 *
 * When using a Quantity from another part of the code, e.g. an imported unit
 * system, use the \c value_as free function rather than \c .value() in order
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
 * The label is used solely for diagnostic purposes.
 *
 * \note The Quantity is designed to be a simple "strong type" class, not a
 * complex mathematical class. To operate on quantities, you must use
 * `value_as`
 * (to operate within the Quantity's unit system) or `native_value_from` (to
 * operate in the Celeritas native unit system), use the resulting numeric
 * values in your mathematical expressions, then return a new Quantity class
 * with the resulting value and correct type.
 */
template<class UnitT, class ValueT>
class Quantity
{
  public:
    //!@{
    //! \name Type aliases
    using value_type = ValueT;
    using unit_type = UnitT;
    using unit_value_type = decltype(UnitT::value());
    using common_type = decltype(std::declval<value_type>()
                                 * std::declval<unit_value_type>());
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
    template<detail::QConstant QC>
    CELER_CONSTEXPR_FUNCTION Quantity(detail::UnitlessQuantity<QC>) noexcept
        : value_(detail::get_constant<ValueT>(QC))
    {
    }

    //!@{
    //! Access the underlying numeric value, discarding units
    CELER_CONSTEXPR_FUNCTION value_type& value() & noexcept { return value_; }
    CELER_CONSTEXPR_FUNCTION value_type const& value() const& noexcept
    {
        return value_;
    }
    //!@}

    //! Access the underlying data for more efficient loading from memory
    CELER_CONSTEXPR_FUNCTION value_type const* data() const { return &value_; }

  private:
    value_type value_{};
};
//---------------------------------------------------------------------------//

//! Type alias for a quantity that uses compile-time precision
template<class UnitT>
using RealQuantity = Quantity<UnitT, real_type>;

//---------------------------------------------------------------------------//
//! \cond
#define CELER_DEFINE_QUANTITY_CMP(TOKEN)                           \
    template<class U, class T, class T2>                           \
    CELER_CONSTEXPR_FUNCTION bool operator TOKEN(                  \
        Quantity<U, T> lhs, Quantity<U, T2> rhs) noexcept          \
    {                                                              \
        return lhs.value() TOKEN rhs.value();                      \
    }                                                              \
    template<class U, class T, detail::QConstant QC>               \
    CELER_CONSTEXPR_FUNCTION bool operator TOKEN(                  \
        Quantity<U, T> lhs, detail::UnitlessQuantity<QC>) noexcept \
    {                                                              \
        return lhs.value() TOKEN detail::get_constant<T>(QC);      \
    }                                                              \
    template<class U, class T, detail::QConstant QC>               \
    CELER_CONSTEXPR_FUNCTION bool operator TOKEN(                  \
        detail::UnitlessQuantity<QC>, Quantity<U, T> rhs) noexcept \
    {                                                              \
        return detail::get_constant<T>(QC) TOKEN rhs.value();      \
    }                                                              \
    namespace detail                                               \
    {                                                              \
    template<detail::QConstant C1, detail::QConstant C2>           \
    CELER_CONSTEXPR_FUNCTION bool                                  \
    operator TOKEN(detail::UnitlessQuantity<C1>,                   \
                   detail::UnitlessQuantity<C2>) noexcept          \
    {                                                              \
        return static_cast<int>(C1) TOKEN static_cast<int>(C2);    \
    }                                                              \
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
operator+(Quantity<U, T> lhs, Quantity<U, T2> rhs) noexcept
{
    return Quantity<U, std::common_type_t<T, T2>>{lhs.value() + rhs.value()};
}

template<class U, class T, class T2>
CELER_CONSTEXPR_FUNCTION auto
operator-(Quantity<U, T> lhs, Quantity<U, T2> rhs) noexcept
{
    return Quantity<U, std::common_type_t<T, T2>>{lhs.value() - rhs.value()};
}

template<class U, class T, class T2>
CELER_CONSTEXPR_FUNCTION auto
operator/(Quantity<U, T> lhs, Quantity<U, T2> rhs) noexcept
{
    return lhs.value() / rhs.value();
}

template<class U, class T>
CELER_CONSTEXPR_FUNCTION auto operator-(Quantity<U, T> q) noexcept
{
    return Quantity<U, T>{-q.value()};
}

template<class U, class T, class T2>
CELER_CONSTEXPR_FUNCTION auto operator*(Quantity<U, T> lhs, T2 rhs) noexcept
{
    return Quantity<U, std::common_type_t<T, T2>>{lhs.value() * rhs};
}

template<class T, class U, class T2>
CELER_CONSTEXPR_FUNCTION auto operator*(T rhs, Quantity<U, T2> lhs) noexcept
{
    return Quantity<U, std::common_type_t<T, T2>>{rhs * lhs.value()};
}

template<class U, class T, class T2>
CELER_CONSTEXPR_FUNCTION auto operator/(Quantity<U, T> lhs, T2 rhs) noexcept
{
    return Quantity<U, std::common_type_t<T, T2>>{lhs.value() / rhs};
}
//!@!}

//! \endcond
//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Get a typeless zero quantity.
 *
 * The zero quantity can be compared against any Quantity.
 */
CELER_CONSTEXPR_FUNCTION auto zero_quantity() noexcept
{
    return detail::UnitlessQuantity<detail::QConstant::zero>{};
}

//---------------------------------------------------------------------------//
/*!
 * Get a typeless quantitity greater than any other numeric quantity.
 */
CELER_CONSTEXPR_FUNCTION auto max_quantity() noexcept
{
    return detail::UnitlessQuantity<detail::QConstant::max>{};
}

//---------------------------------------------------------------------------//
/*!
 * Get a quantitity less than any other numeric quantity.
 */
CELER_CONSTEXPR_FUNCTION auto neg_max_quantity() noexcept
{
    return detail::UnitlessQuantity<detail::QConstant::neg_max>{};
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
native_value_from(Quantity<UnitT, ValueT> quant) noexcept
{
    using common_type = typename Quantity<UnitT, ValueT>::common_type;
    return static_cast<common_type>(quant.value())
           * static_cast<common_type>(UnitT::value());
}

//---------------------------------------------------------------------------//
/*!
 * Create a quantity from a value in the Celeritas unit system.
 *
 * This function can be used for defining a constant for use in another unit
 * system (typically a "natural" unit system for use in physics kernels).
 *
 * An extra cast may be needed when mixing \c float, \c double, and \c
 * celeritas::Constant.
 */
template<class Q, class T>
CELER_CONSTEXPR_FUNCTION Q native_value_to(T value) noexcept
{
    using common_type = typename Q::common_type;
    using value_type = typename Q::value_type;
    constexpr auto unit_value = Q::unit_type::value();
    return Q{
        static_cast<value_type>(value / static_cast<common_type>(unit_value))};
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
   cout << accessor_unit_label<decltype(&ParticleView::mass)>() << endl;
   \endcode
 */
template<class T>
inline char const* accessor_unit_label()
{
    return detail::AccessorResultType<T>::unit_type::label();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
