//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/detail/QuantityImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "corecel/Macros.hh"
#include "corecel/math/NumericLimits.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
template<class UnitT, class ValueT>
class Quantity;

namespace detail
{
//---------------------------------------------------------------------------//
//! Helper tag for special unitless values
enum class QConstant
{
    neg_max = -1,
    zero = 0,
    max = 1
};

//! Convert unitless values into a particular type
template<class T>
CELER_CONSTEXPR_FUNCTION T get_constant(QConstant qc)
{
    if constexpr (std::is_floating_point_v<T>)
    {
        // Return +/- infinity
        return qc == QConstant::neg_max ? -numeric_limits<T>::infinity()
               : qc == QConstant::max   ? numeric_limits<T>::infinity()
                                        : 0;
    }
    else
    {
        // Return lowest and highest values
        return qc == QConstant::neg_max ? numeric_limits<T>::lowest()
               : qc == QConstant::max   ? numeric_limits<T>::max()
                                        : 0;
    }
}

//! Tag class for creating a nonnumeric value comparable to Quantity.
template<QConstant QC>
struct UnitlessQuantity
{
};

//---------------------------------------------------------------------------//
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
//! Template matching to determine if T is a Quantity
template<class T>
struct IsQuantity : std::false_type
{
};
template<class V, class S>
struct IsQuantity<Quantity<V, S>> : std::true_type
{
};
template<class V, class S>
struct IsQuantity<Quantity<V, S> const> : std::true_type
{
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
