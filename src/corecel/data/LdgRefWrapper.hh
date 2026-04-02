//------------------------------ -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/LdgRefWrapper.hh
//! \sa corecel/data/Ldg.test.cc
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include <type_traits>

#include "corecel/Macros.hh"

#include "LdgTraits.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Wrap the low-level CUDA/HIP "load read-only global memory" function.
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
 * Reference wrapper that loads data safely via \c ldg on conversion.
 *
 * Like \c std::reference_wrapper, this stores a pointer to a const object
 * and provides an implicit conversion to the value type. However, the value
 * being wrapped \em must be a const reference, and the return is a \c value
 * rather than a reference .
 *
 * The \c __ldg intrinsic is invoked during the implicit load, so client code
 * can bind the wrapper to an ordinary value variable transparently.
 */
template<class T>
class LdgRefWrapper
{
    static_assert(std::is_const_v<T>);
    static_assert(is_ldg_supported_v<std::remove_const_t<T>>,
                  "const arithmetic, OpaqueId or enum type required");

  public:
    //!@{
    //! \name Type aliases
    using type = std::remove_const_t<T>;
    //!@}

  public:
    //! Construct from a const reference to the target
    CELER_CEF LdgRefWrapper(T& ref) noexcept : ptr_{&ref} {}

    //! Load the referenced value using __ldg
    CELER_CEF type get() const noexcept { return ldg(ptr_); }

    //! Implicit conversion: load via __ldg
    CELER_CEF operator type() const noexcept { return this->get(); }

    //!@{
    /*!
     * Comparison operators against the underlying type.
     *
     * Defined here so template \c operator== (e.g. \c OpaqueId) are found via
     * ADL without requiring implicit conversion during deduction.
     */
    CELER_CEF friend bool operator==(LdgRefWrapper a, type b) noexcept
    {
        return a.get() == b;
    }
    CELER_CEF friend bool operator==(type a, LdgRefWrapper b) noexcept
    {
        return a == b.get();
    }
    CELER_CEF friend bool operator!=(LdgRefWrapper a, type b) noexcept
    {
        return a.get() != b;
    }
    CELER_CEF friend bool operator!=(type a, LdgRefWrapper b) noexcept
    {
        return a != b.get();
    }
    //!@}

  private:
    T* ptr_;
};

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
