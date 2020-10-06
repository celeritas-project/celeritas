//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Quantity.hh
//---------------------------------------------------------------------------//
#pragma once

#include "Macros.hh"
#include "Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Tag class for creating a quantity implicitly from a zero value
struct ZeroQuantity
{
};

//! Get a zero quantity (analogous to nullptr)
CELER_CONSTEXPR_FUNCTION ZeroQuantity zero_quantity()
{
    return ZeroQuantity{};
}

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
   real_type eval = energy.value(); // Natural units
   MevMomentum momentum{std::sqrt(eval * eval + 2 * mass.value() * eval)};
   \endcode
 * The resulting quantity can be converted to the native Celeritas unit system
 * with a `unit_cast`, which multiplies in the constant value of
 * ElMomentumUnit:
 * \code
 * real_type mom = unit_cast(momentum);
 * \endcode
 *
 * \note The Quantity is designed to be a simple "strong type" class, not a
 * complex mathematical class. To operate on quantities, you must use `value()`
 * or `unit_cast` as appropriate and operate on the numeric values as
 * appropriate, then return a new Quantity class as appropriate.
 */
template<class UnitT, class ValueT = real_type>
class Quantity
{
  public:
    //@{
    //! Type aliases
    using value_type = ValueT;
    using unit_type  = UnitT;
    //@}

  public:
    //! Construct with default (zero)
    CELER_CONSTEXPR_FUNCTION Quantity() : value_(0) {}

    //! Construct with value in celeritas native units
    explicit CELER_CONSTEXPR_FUNCTION Quantity(value_type value)
        : value_(value)
    {
    }

    //! Construct implicitly from a zero
    CELER_CONSTEXPR_FUNCTION Quantity(ZeroQuantity) : value_(0) {}

    //! Get numeric value, discarding units.
    CELER_CONSTEXPR_FUNCTION value_type value() const { return value_; }

  private:
    value_type value_;
};

//---------------------------------------------------------------------------//
// COMPARATORS
//---------------------------------------------------------------------------//
#define CELER_DEFINE_QUANTITY_CMP(TOKEN)                             \
    template<class U, class T>                                       \
    CELER_CONSTEXPR_FUNCTION bool operator TOKEN(Quantity<U, T> lhs, \
                                                 Quantity<U, T> rhs) \
    {                                                                \
        return lhs.value() TOKEN rhs.value();                        \
    }                                                                \
    template<class U, class T>                                       \
    CELER_CONSTEXPR_FUNCTION bool operator TOKEN(Quantity<U, T> lhs, \
                                                 ZeroQuantity)       \
    {                                                                \
        return lhs.value() TOKEN T{0};                               \
    }                                                                \
    template<class U, class T>                                       \
    CELER_CONSTEXPR_FUNCTION bool operator TOKEN(ZeroQuantity,       \
                                                 Quantity<U, T> rhs) \
    {                                                                \
        return T{0} TOKEN rhs.value();                               \
    }

//@{
//! Comparisons for Quantity
CELER_DEFINE_QUANTITY_CMP(==)
CELER_DEFINE_QUANTITY_CMP(!=)
CELER_DEFINE_QUANTITY_CMP(<)
CELER_DEFINE_QUANTITY_CMP(>)
CELER_DEFINE_QUANTITY_CMP(<=)
CELER_DEFINE_QUANTITY_CMP(>=)
//@}

#undef CELER_DEFINE_QUANTITY_CMP

//---------------------------------------------------------------------------//
// UNIT HELPERS
//---------------------------------------------------------------------------//
//! Value is C1::value() / C2::value()
template<class C1, class C2>
struct UnitDivide
{
    static CELER_CONSTEXPR_FUNCTION real_type value()
    {
        return C1::value() / C2::value();
    }
};

//! Value is C1::value() * C2::value()
template<class C1, class C2>
struct UnitProduct
{
    static CELER_CONSTEXPR_FUNCTION real_type value()
    {
        return C1::value() * C2::value();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Convert the given quantity into the native celeritas unit system.
 *
 * \code
 assert(unit_cast(Quantity<real_type, SpeedOfLight>{1})
        == 2.998e10 * centimeter/second);
 * \endcode
 */
template<class UnitT, class ValueT>
CELER_CONSTEXPR_FUNCTION auto unit_cast(Quantity<UnitT, ValueT> quant)
    -> decltype(ValueT{} * UnitT::value())
{
    return quant.value() * UnitT::value();
}

//---------------------------------------------------------------------------//
} // namespace celeritas
