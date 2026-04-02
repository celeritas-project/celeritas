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

#if !CELER_DEVICE_COMPILE
#    include <ostream>
#endif

namespace celeritas
{
class Constant;
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
   real_type m_mev = value_as<MevMass>(mass); // Natural units
   MevMomentum momentum{sqrt(ipow<2>(e_mev) + 2 * m_mev * e_mev))};
   \endcode
 * The resulting quantity can be converted to the native Celeritas unit system
 * with `native_value_from`, which multiplies in the constant value of
 * ElMomentumUnit:
 * \code
   real_type mom = native_value_from(momentum);
 * \endcode
 * Conversion from a native unit to a quantity requires the template argument:
 * \code
   auto amu = native_value_to<MevMass>(constants::atomic_mass);
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
 * Furthermore, this class is \em not compatible with \c Array.
 */
template<class UnitT, class ValueT>
class Quantity
{
    static_assert(std::is_arithmetic_v<ValueT>,
                  "value type must be arithmetic");
    static_assert(std::is_arithmetic_v<decltype(UnitT::value())>
                      || std::is_same_v<decltype(UnitT::value()), Constant>,
                  "unit value type must be arithmetic or constant");

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

    //! Construct with quantity of same unit but different value type
    template<class ValueT2,
             std::enable_if_t<!std::is_same_v<ValueT, ValueT2>
                                  && std::is_convertible_v<ValueT2, ValueT>,
                              int>
             = 0>
    CELER_CONSTEXPR_FUNCTION Quantity(Quantity<UnitT, ValueT2> other) noexcept
        : value_(static_cast<ValueT>(other.value()))
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

    //// INLINE COMPARATOR FRIENDS ////

#define CELER_DEFINE_QUANTITY_CMP(TOKEN)                           \
    template<class T2>                                             \
    CELER_CONSTEXPR_FUNCTION friend bool operator TOKEN(           \
        Quantity lhs, Quantity<UnitT, T2> rhs) noexcept            \
    {                                                              \
        return lhs.value() TOKEN rhs.value();                      \
    }                                                              \
    template<detail::QConstant QC>                                 \
    CELER_CONSTEXPR_FUNCTION friend bool operator TOKEN(           \
        Quantity lhs, detail::UnitlessQuantity<QC>) noexcept       \
    {                                                              \
        return lhs.value() TOKEN detail::get_constant<ValueT>(QC); \
    }                                                              \
    template<detail::QConstant QC>                                 \
    CELER_CONSTEXPR_FUNCTION friend bool operator TOKEN(           \
        detail::UnitlessQuantity<QC>, Quantity rhs) noexcept       \
    {                                                              \
        return detail::get_constant<ValueT>(QC) TOKEN rhs.value(); \
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

  private:
    template<class T2>
    using OtherQuantity = Quantity<UnitT, std::common_type_t<ValueT, T2>>;

  public:
    //// INLINE OPERATOR FRIENDS ////

    //!@{
    //! Arithmetic with unitless scalars

    template<class T2>
    CELER_CONSTEXPR_FUNCTION friend auto
    operator*(Quantity lhs, T2 rhs) noexcept
    {
        return OtherQuantity<T2>{lhs.value() * rhs};
    }

    template<class T2>
    CELER_CONSTEXPR_FUNCTION friend auto
    operator*(T2 lhs, Quantity rhs) noexcept
    {
        return OtherQuantity<T2>{lhs * rhs.value()};
    }

    template<class T2>
    CELER_CONSTEXPR_FUNCTION friend auto
    operator/(Quantity lhs, T2 rhs) noexcept
    {
        return OtherQuantity<T2>{lhs.value() / rhs};
    }

    //!@}

    //!@{
    //! Operators with same units
    template<class T2>
    CELER_CONSTEXPR_FUNCTION friend auto
    operator+(Quantity lhs, Quantity<UnitT, T2> rhs) noexcept
    {
        return OtherQuantity<T2>{lhs.value() + rhs.value()};
    }

    template<class T2>
    CELER_CONSTEXPR_FUNCTION friend auto
    operator-(Quantity lhs, Quantity<UnitT, T2> rhs) noexcept
    {
        return OtherQuantity<T2>{lhs.value() - rhs.value()};
    }

    template<class T2>
    CELER_CONSTEXPR_FUNCTION friend auto
    operator/(Quantity lhs, Quantity<UnitT, T2> rhs) noexcept
    {
        return lhs.value() / rhs.value();
    }

    //!@}

    CELER_CONSTEXPR_FUNCTION friend auto operator-(Quantity q) noexcept
    {
        return Quantity{-q.value()};
    }

  private:
    value_type value_{};
};
//---------------------------------------------------------------------------//

//! Type alias for a quantity that uses compile-time precision
template<class UnitT>
using RealQuantity = Quantity<UnitT, real_type>;

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
 * Get a typeless quantity greater than any other numeric quantity.
 */
CELER_CONSTEXPR_FUNCTION auto max_quantity() noexcept
{
    return detail::UnitlessQuantity<detail::QConstant::max>{};
}

//---------------------------------------------------------------------------//
/*!
 * Get a quantity less than any other numeric quantity.
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

#if !CELER_DEVICE_COMPILE
//---------------------------------------------------------------------------//
/*!
 * Output a quantity with its label.
 */
template<class UnitT, class ValueT>
std::ostream& operator<<(std::ostream& os, Quantity<UnitT, ValueT> const& q)
{
    static_assert(sizeof(UnitT::label()) > 0,
                  "Unit does not have a 'label' definition");
    os << q.value() << " [" << UnitT::label() << ']';
    return os;
}

#endif

//---------------------------------------------------------------------------//
//! True if T is a Quantity
template<class T>
inline constexpr bool is_quantity_v = detail::IsQuantity<T>::value;

//---------------------------------------------------------------------------//
template<class T, class>
struct LdgTraits;

// Set up cached const global loading for Quantity
template<class U, class T>
struct LdgTraits<Quantity<U, T>, void>
{
    using underlying_type = typename Quantity<U, T>::value_type;

    static CELER_CONSTEXPR_FUNCTION underlying_type const*
    data(Quantity<U, T> const* ptr)
    {
        return ptr->data();
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
