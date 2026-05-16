//------------------------------ -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/Ldg.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "corecel/Macros.hh"
#if CELER_DEVICE_COMPILE
// Make sure __ldg is available for HIP
#    include "corecel/DeviceRuntimeApi.hh"
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * \page ldg Cached device loading
 *
 * On GPUs, reading from global memory through the L1/texture cache can improve
 * throughput when many threads access the same address.
 * The \c __ldg intrinsic (CUDA/HIP) performs such a cached load, using
 * L1/texture memory rather than the ordinary data cache.
 * The hardware contract is that the pointed-to memory must be \em read-only
 * for the lifetime of the kernel; this is generally true for \em Params data
 * (physics tables, geometry) but not for \em State data.
 * On the host the \c ldg family of functions falls back to a plain
 * dereference, so no special-casing is needed in caller code.
 * Because \c ldg is only for \em read-only addresses in \em global memory, all
 * arguments \em must match only \c const_pointer types.
 *
 * \par Scalar values
 *
 * Pass a const pointer to any supported type to the one-argument \c ldg:
 * \code
 *   real_type energy = ldg(&record.energy);
 *   MaterialId mat   = ldg(&record.material);  // OpaqueId supported
 * \endcode
 *
 * Built-in implementations cover:
 *
 * - arithmetic types (identity, the default),
 * - enum types (reinterpret-cast to the underlying integer),
 * - \c OpaqueId (pointer to the underlying index \c T),
 * - \c Quantity (pointer to the underlying value \c T),
 * - \c Array (successive loads to each element).
 *
 * \par Spans and collections
 *
 * \c LdgSpan<T const> (from \c corecel/cont/LdgSpan.hh) is an alias for
 * \c Span whose iterator triggers \c __ldg on every element access.
 * Use it as you would any ordinary span:
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
 * The \c ldg function is found by argument-dependent lookup (ADL).
 * Implement a \c friend function \c ldg that takes a const pointer and returns
 * a value:
 * \code
 *   template<class T>
 *   class MyClass
 *   {
 *     public:
 *       CELER_CONSTEXPR_FUNCTION friend MyClass ldg(MyClass const* m)
 *       {
 *         return MyClass{ldg(&m->value_)};
 *       }
 *
 *     private:
 *       T value_;
 *   };
 * \endcode
 * See \c OpaqueId, \c Quantity and \c Array for examples.
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
 * \c operator* returns an \c LdgWrapper.
 * Wrapping it in \c Span yields \c LdgSpan: range-for loops and standard
 * algorithms transparently trigger \c __ldg on every element access without
 * requiring any change at the call site.
 */

//---------------------------------------------------------------------------//
/*!
 * Wrap the low-level CUDA/HIP "load read-only global memory" function.
 *
 * On CUDA the load is cached in L1/texture memory, improving performance when
 * data is repeatedly read by many threads in a kernel.
 *
 * \warning The target must be \c global memory (or else the kernel may crash)
 * and \c read-only (or else the result may be wrong) for the lifetime of the
 * kernel.
 * This is generally true for Params data but not State data.
 */
template<class T>
CELER_CONSTEXPR_FUNCTION std::enable_if_t<std::is_arithmetic_v<T>, T>
ldg(T const* ptr) noexcept
{
#if CELER_DEVICE_COMPILE
    return __ldg(ptr);
#else
    return *ptr;
#endif
}

//---------------------------------------------------------------------------//
//! \cond (CELERITAS_DOC_DEV)
/*!
 * Get a pointer to the underlying integer for an enum type.
 */
template<class T>
CELER_CONSTEXPR_FUNCTION std::enable_if_t<std::is_enum_v<T>, T>
ldg(T const* ptr) noexcept
{
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    auto const* data = reinterpret_cast<std::underlying_type_t<T> const*>(ptr);
    return static_cast<T>(ldg(data));
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
CELER_CONSTEXPR_FUNCTION T ldg(Class const& obj, T Class::* mp) noexcept
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

    CELER_CONSTEXPR_FUNCTION T operator()(Class const& obj) const
    {
        return ldg(&(obj.*mp));
    }
};

//! Deduction guide: \c LdgMember{&Foo::bar} deduces \c LdgMember<Foo,Bar>
template<class Class, class T>
LdgMember(T Class::*) -> LdgMember<Class, T>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
