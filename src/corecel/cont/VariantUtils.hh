//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/cont/VariantUtils.hh
//! \brief Host-only utilities for use with std::variant
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>
#include <utility>

#include "detail/VariantUtilsImpl.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Helper class for dispatching type-specific lambdas.
 *
 * Example applied to a variant that converts to int or string: \code
  std::visit(Overload{[](int a) { cout << a + 2; },
                      [](std::string const& s) { cout << '"' << s << '"'; }},
             my_variant);
 * \endcode
 */
template<typename... Ts>
struct Overload : Ts...
{
    using Ts::operator()...;
};

// Template deduction guide
template<class... Ts>
Overload(Ts&&...) -> Overload<Ts...>;

//---------------------------------------------------------------------------//
/*!
 * Create a wrapper functor for unifying the return type.
 *
 * This provides a unified return type \c T (usually a variant) that can be
 * implicitly constructed from all return types of a functor \c F that operates
 * on a generic type \c U . The class is necessary because \c std::visit
 * requires all return types to be the same.
 */
template<class T, class F>
detail::ReturnAsImpl<T, F> return_as(F&& func)
{
    return {std::forward<F>(func)};
}

//---------------------------------------------------------------------------//
/*!
 * Define a variant that contains all the classes mapped by an enum+traits.
 *
 * For example: \code
    using VariantSurface = EnumVariant<SurfaceType, SurfaceTypeTraits>;
 * \endcode
 * is equivalent to: \code
    using VariantSurface = std::variant<PlaneX, PlaneY, ..., GeneralQuadric>;
 * \endcode
 *
 * \sa EnumClassUtils.hh
 */
template<class E, template<E> class ETraits>
using EnumVariant = typename detail::EnumVariantImpl<E, ETraits>::type;

//---------------------------------------------------------------------------//
}  // namespace celeritas
