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
 * \page ldg Cached device loading
 *
 * On GPUs, reading from global memory through the L1/texture cache can improve
 * throughput when many threads access the same address. The \c __ldg
 * intrinsic (CUDA/HIP) performs such a cached load, using L1/texture memory
 * rather than the ordinary data cache. The hardware contract is that the
 * pointed-to memory must be \em read-only for the lifetime of the kernel;
 * this is generally true for \em Params data (physics tables, geometry) but
 * not for \em State data. On the host the \c ldg family of functions falls
 * back to a plain dereference, so no special-casing is needed in caller code.
 * Because \c ldg is only for read-only addresses, all arguments \em must
 * match only \c const types.
 *
 * \par Scalar values
 *
 * Pass a const pointer to any supported type to the one-argument \c ldg:
 * \code
 *   real_type energy = ldg(&record.energy);
 *   MaterialId mat   = ldg(&record.material);  // OpaqueId supported
 * \endcode
 *
 * \par Struct members
 *
 * Load a single member without reading the whole struct using the
 * two-argument overload or the storable \c LdgMember projector:
 * \code
 *   // Immediate two-argument form
 *   BIHNodeId parent = ldg(node, &BIHLeafNode::parent);
 *
 *   // Storable callable -- useful with algorithms
 *   auto load_parent = LdgMember{&BIHLeafNode::parent};
 *   BIHNodeId parent = load_parent(node);
 * \endcode
 *
 * \par Spans and collections
 *
 * \c LdgSpan<T const> (from \c corecel/cont/LdgSpan.hh) is an alias for
 * \c Span whose iterator triggers \c __ldg on every element access. Use it
 * as you would any ordinary span:
 * \code
 *   LdgSpan<real_type const> energies = params.get_energies();
 *   for (real_type e : energies)   // each read uses __ldg
 *       process(e);
 * \endcode
 *
 * \c Collection<T, Ownership::const_reference, MemSpace::device> returns
 * \c LdgSpan automatically when the element type supports \c ldg, so View
 * classes built on device \c const_reference collections benefit without any
 * extra work.
 *
 * \par Extending ldg to a new type
 *
 * \c ldg dispatches through the customization point \c ldg_data, found by
 * argument-dependent lookup (ADL). To support a new type, define a free
 * function in its namespace that returns a \c const pointer to an arithmetic
 * type. For a wrapper struct holding a single \c int member:
 * \code
 *   namespace myns
 *   {
 *   struct MyCount { int value; };
 *
 *   CELER_CONSTEXPR_FUNCTION int const* ldg_data(MyCount const* p) noexcept
 *   {
 *       return &p->value;
 *   }
 *   }  // namespace myns
 * \endcode
 *
 * Built-in overloads cover:
 * - arithmetic types (identity, the default),
 * - enum types (reinterpret-cast to the underlying integer),
 * - \c OpaqueId<I,T> (pointer to the underlying index \c T), and
 * - \c Quantity<U,T> (pointer to the underlying value \c T).
 *
 * \internal
 *
 * \par Implementation details
 *
 * \c detail::LdgWrapper<T const> is a thin proxy (similar to
 * \c std::reference_wrapper) that stores a \c const pointer and implicitly
 * converts to the value type by calling \c ldg. The result is always a
 * \em value, not a reference, and the load goes through \c __ldg on device.
 *
 * \c detail::LdgIterator<T const> is a random-access iterator whose
 * \c operator* returns an \c LdgWrapper. Wrapping it in \c Span yields
 * \c LdgSpan: range-for loops and standard algorithms transparently trigger
 * \c __ldg on every element access without requiring any change at the
 * call site.
 */

//---------------------------------------------------------------------------//
/*!
 * Get a pointer to the arithmetic data for use with \c __ldg .
 *
 * Default overload for arithmetic types: returns the pointer unchanged.
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
 * This relies on \c ldg_data found by ADL to obtain a pointer to the
 * underlying arithmetic type; see \ref ldg for usage and extension examples.
 *
 * On CUDA the load is cached in L1/texture memory, improving performance when
 * data is repeatedly read by many threads in a kernel.
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
//! \cond (CELERITAS_DOC_DEV)
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

//! \endcond
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
