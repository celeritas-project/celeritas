//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/ArrayQuantity.hh
//! \brief Create and convert arrays of quantities
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include <type_traits>

#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"

#include "Quantity.hh"

namespace celeritas
{

//---------------------------------------------------------------------------//
/*!
 * Construct an array of quantities from raw values.
 *
 * This helper function allows concise construction of arrays of quantities:
 * \code
 * auto distances = make_quantity_array<CmLength>(1.0, 2.5, 3.7);
 * \endcode
 */
template<class Q, class... Args>
CELER_CONSTEXPR_FUNCTION Array<Q, sizeof...(Args)>
make_quantity_array(Args const&... args) noexcept
{
    static_assert(is_quantity_v<Q>);
    return {Q{args}...};
}

//! \cond (CELERITAS_DOC_DEV)
//---------------------------------------------------------------------------//
/*!
 * Construct an array of quantities from raw values.
 *
 * This helper function allows concise construction of arrays of quantities:
 * \code
 * auto pos = make_quantity_array<CmLength>(hardcoded_pos_cm);
 * \endcode
 */
template<class Q, size_type N>
CELER_CONSTEXPR_FUNCTION Array<Q, N>
make_quantity_array(Array<typename Q::value_type, N> const& arr) noexcept
{
    static_assert(is_quantity_v<Q>);
    Array<Q, N> result;
    for (size_type i = 0; i < N; ++i)
    {
        result[i] = Q{arr[i]};
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Convert an array of quantities to native values.
 *
 * This applies native_value_from element-wise to each component.
 */
template<class UnitT, class ValueT, size_type N>
CELER_CONSTEXPR_FUNCTION auto
native_value_from(Array<Quantity<UnitT, ValueT>, N> const& quant) noexcept
{
    using common_type = typename Quantity<UnitT, ValueT>::common_type;
    Array<common_type, N> result;
    for (size_type i = 0; i < N; ++i)
    {
        result[i] = native_value_from(quant[i]);
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Convert an array of native values to an array of quantities.
 *
 * This applies native_value_to element-wise to each component.
 */
template<class Q, class T, size_type N>
CELER_CONSTEXPR_FUNCTION auto native_value_to(Array<T, N> const& value) noexcept
{
    static_assert(is_quantity_v<Q>);
    Array<Q, N> result;
    for (size_type i = 0; i < N; ++i)
    {
        result[i] = native_value_to<Q>(value[i]);
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get the values from an array of quantities.
 *
 * This applies value_as element-wise to each component.
 */
template<class Q, size_type N>
CELER_CONSTEXPR_FUNCTION auto value_as(Array<Q, N> const& quant) noexcept
    -> std::enable_if_t<is_quantity_v<Q>, Array<typename Q::value_type, N>>
{
    Array<typename Q::value_type, N> result;
    for (size_type i = 0; i < N; ++i)
    {
        result[i] = value_as<Q>(quant[i]);
    }
    return result;
}

//! \endcond
//---------------------------------------------------------------------------//
}  // namespace celeritas
