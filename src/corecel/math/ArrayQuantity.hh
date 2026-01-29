//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/ArrayQuantity.hh
//! \brief Create and convert arrays of quantities
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>

#include "corecel/cont/Array.hh"
#include "corecel/math/Quantity.hh"

namespace celeritas
{

//---------------------------------------------------------------------------//
/*!
 * Construct an array of quantities from raw values.
 *
 * This helper function allows concise construction of quantity arrays:
 * \code
 * auto distances = make_quantity_array<CmLength>(1.0, 2.5, 3.7);
 * \endcode
 */
template<class Q, class... Args>
CELER_CONSTEXPR_FUNCTION Array<Q, sizeof...(Args)>
make_quantity_array(Args const&... args) noexcept
{
    return {Q{args}...};
}

//---------------------------------------------------------------------------//
/*!
 * Convert an array of quantities to native values.
 *
 * This applies native_value_from element-wise to each component.
 */
template<class UnitT, class ValueT, std::size_t N>
CELER_CONSTEXPR_FUNCTION auto
native_value_from(Array<Quantity<UnitT, ValueT>, N> const& quant) noexcept
{
    using common_type = typename Quantity<UnitT, ValueT>::common_type;
    Array<common_type, N> result;
    for (std::size_t i = 0; i < N; ++i)
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
template<class Q, class T, std::size_t N>
CELER_CONSTEXPR_FUNCTION auto native_value_to(Array<T, N> const& value) noexcept
{
    Array<Q, N> result;
    for (std::size_t i = 0; i < N; ++i)
    {
        result[i] = native_value_to<Q>(value[i]);
    }
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
