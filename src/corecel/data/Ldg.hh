//------------------------------ -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/Ldg.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include <type_traits>

#include "corecel/Macros.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * \page ldg Cached global loading with __ldg wrappers
 */

//---------------------------------------------------------------------------//
/*!
 * Get a pointer to the arithmetic data for use with \c __ldg .
 *
 * Default overload for arithmetic types: returns the pointer unchanged.
 *
 * To extend \c ldg support to a new type, define a free function
 * \c ldg_data(MyType const*) in the namespace of \c MyType (enabling
 * ADL-based lookup). The function must return a \c const pointer to an
 * arithmetic type.
 */
template<class T>
CELER_CONSTEXPR_FUNCTION std::enable_if_t<std::is_arithmetic_v<T>, T const*>
ldg_data(T const* ptr) noexcept
{
    return ptr;
}

//---------------------------------------------------------------------------//
/*!
 * Get a pointer to the underlying integer for an enum type.
 */
template<class T>
CELER_CONSTEXPR_FUNCTION
    std::enable_if_t<std::is_enum_v<T>, std::underlying_type_t<T> const*>
    ldg_data(T const* ptr) noexcept
{
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    return reinterpret_cast<std::underlying_type_t<T> const*>(ptr);
}

//---------------------------------------------------------------------------//
/*!
 * Wrap the low-level CUDA/HIP "load read-only global memory" function.
 *
 * This relies on a default implementation of \c ldg_data that allows user
 * overrides via ADL. To extend this functionality, provide an overload in your
 class's namespace that returns a const pointer to an arithmetic type. <insert
 example here>
 *
 * This low-level capability allows improved caching because we're \em
 * promising that the data is not mem. For CUDA the load is cached in
 * L1/texture memory, theoretically improving performance if repeatedly
 * accessed.
 *
 * \warning The target address must be read-only for the lifetime of the
 * kernel. This is generally true for Params data but not State data.
 */
template<class T>
CELER_CONSTEXPR_FUNCTION T ldg(T const* ptr)
{
    auto const* data_ptr = ldg_data(ptr);
    using data_type
        = std::remove_cv_t<std::remove_pointer_t<decltype(data_ptr)>>;
    static_assert(std::is_arithmetic_v<data_type>,
                  R"(Only arithmetic-underlying types are supported by __ldg)");

#if CELER_DEVICE_COMPILE
    return T{__ldg(data_ptr)};
#else
    return T{*data_ptr};
#endif
}

//---------------------------------------------------------------------------//
/*!
 * Load a struct member via \c ldg using a pointer-to-member.
 *
 * Convenience overload for when the member is known at the call site.
 * \code
 * BIHNodeId parent = ldg(node, &BIHLeafNode::parent);
 * \endcode
 */
template<class Class, class T>
CELER_FUNCTION T ldg(Class const& obj, T Class::* mp)
{
    return ldg(&(obj.*mp));
}

//---------------------------------------------------------------------------//
/*!
 * Storable projector that loads a struct member via \c ldg .
 *
 * Stores a pointer-to-member and, when called with an object, returns the
 * member value loaded via \c __ldg . Use this when the load must be captured
 * as a callable; for immediate use prefer the two-argument \c ldg overload.
 *
 * \code
 * auto load_parent = LdgMember{&BIHLeafNode::parent};
 * BIHNodeId parent = load_parent(node);
 * \endcode
 */
template<class Class, class T>
struct LdgMember
{
    T Class::* mp;

    CELER_FUNCTION T operator()(Class const& obj) const
    {
        return ldg(&(obj.*mp));
    }
};

//! Deduction guide: \c LdgMember{&Foo::bar} deduces \c LdgMember<Foo,Bar>
template<class Class, class T>
LdgMember(T Class::*) -> LdgMember<Class, T>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
