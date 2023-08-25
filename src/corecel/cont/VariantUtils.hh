//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/cont/VariantUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include <utility>

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Implementation of \c return_as.
 */
template<class T, class F>
struct ReturnAsImpl
{
    F apply;

    template<class U>
    T operator()(U&& other)
    {
        return this->apply(std::forward<U>(other));
    }
};
//---------------------------------------------------------------------------//
}  // namespace detail

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
}  // namespace celeritas
