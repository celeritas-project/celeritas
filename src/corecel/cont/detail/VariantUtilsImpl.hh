//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/cont/detail/VariantUtilsImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>
#include <utility>
#include <variant>

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
/*!
 * Implementation for a variant that uses traits based on an enum.
 */
template<class E, template<E> class ETraits>
struct EnumVariantImpl
{
    //! Underlying integer for enum
    using EInt = std::underlying_type_t<E>;

    static_assert(static_cast<EInt>(E::size_) > 0, "ill-defined enumeration");

    //! Compile-time range of all allowable enum integer values
    using EIntRange
        = std::make_integer_sequence<EInt, static_cast<EInt>(E::size_)>;

    //! Map enum integer value to class
    template<EInt I>
    struct EIntTraits
    {
        using type = typename ETraits<static_cast<E>(I)>::type;
    };

    //!@{
    //! Expand an integer sequence into a variant
    template<class Seq>
    struct VariantImpl;
    template<EInt... Is>
    struct VariantImpl<std::integer_sequence<EInt, Is...>>
    {
        using type = std::variant<typename EIntTraits<Is>::type...>;
    };
    //!@}

    //! Get a variant with all the surface types
    using type = typename VariantImpl<EIntRange>::type;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
