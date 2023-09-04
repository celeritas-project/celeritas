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

#include "corecel/Assert.hh"

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
/*!
 * Visit a container's element by calling "visit" on the corresponding index.
 *
 * example: \code
   std::vector<std::variant<int, stdd:string>> myvec{"hi", 123, "bye"};
   ContainerVisitor visit_element{myvec};
   visit_element([](auto&& v) { cout << v; }, 1); // Prints '123'
   visit_element([](auto&& v) { cout << v; }, 2); // Prints 'bye'
   \endcode
 */
template<class T, class U = typename std::remove_reference_t<T>::size_type>
struct ContainerVisitor
{
    using index_type = U;

    T container;

    template<class F>
    decltype(auto) operator()(F&& func, U const& idx) const
    {
        auto&& value = container[idx];
        CELER_ASSUME(!value.valueless_by_exception());
        return std::visit(std::forward<F>(func), std::move(value));
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
