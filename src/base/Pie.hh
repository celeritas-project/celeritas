//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Collection.hh
//---------------------------------------------------------------------------//
#pragma once

#include "OpaqueId.hh"
#include "CollectionTypes.hh"
#include "Range.hh"
#include "Types.hh"
#include "detail/CollectionImpl.hh"

namespace celeritas
{
//! Opaque ID representing a single element of a container.
template<class T>
using ItemId = OpaqueId<T, unsigned int>;

//---------------------------------------------------------------------------//
/*!
 * Reference a contiguous range of IDs corresponding to a slice of items.
 *
 * \tparam T The value type of items to represent.
 *
 * A ItemRange is a range of \c OpaqueId<T> that reference a range of values of
 * type \c T in a \c Collection . The ItemRange acts like a \c slice object in
 * Python when used on a Collection, returning a Span<T> of the underlying
 * data.
 *
 * A ItemRange is only meaningful in connection with a particular Collection of
 * type T. It doesn't have any persistent connection to its associated
 * collection and thus must be used carefully.
 *
 * \todo It might also be good to have a `CollectionMap` -- mapping one
 * OpaqueId to another OpaqueId type (with just an offset value). This would be
 * used for example in physics, where \c ItemRange objects themselves are
 * supposed to be indexed into with a particular ID type.
 *
 * \code
 * struct MyMaterial
 * {
 *     real_type number_density;
 *     ItemRange<ElementComponents> components;
 * };
 *
 * template<Ownership W, MemSpace M>
 * struct MyData
 * {
 *     Collection<ElementComponents, W, M> components;
 *     Collection<MyMaterial, W, M> materials;
 * };
 * \endcode
 */
template<class T, class Size = unsigned int>
using ItemRange = Range<OpaqueId<T, Size>>;

//---------------------------------------------------------------------------//
/*!
 * Manage generic array-like data ownership and transfer from host to device.
 *
 * Data are constructed incrementally on the host, then copied (along with
 * their associated ItemRange ) to device. A Collection can act as a
 * std::vector<T>, DeviceVector<T>, Span<T>, or Span<const T>. The Spans can
 * point to host or device memory, but the MemSpace template argument protects
 * against accidental accesses from the wrong memory space.
 *
 * Each Collection object is usually accessed with a Slice, which references a
 * contiguous set of elements in the Collection. For example, setup code on the
 * host would extend the Collection with a series of vectors, the addition of
 * which returns a ItemRange that returns the equivalent data on host or
 * device. This methodology allows complex nested data structures to be built
 * up quickly at setup time without knowing the size requirements beforehand.
 *
 * Host-device functions and classes should use \c Collection with a reference
 * or const_reference Ownership, and the \c MemSpace::native type, which
 * expects device memory when compiled inside a CUDA file and host memory when
 * used inside a C++ source or test. (This design choice prevents a single CUDA
 * file from compiling separate host-compatible and device-compatible compute
 * kernels, but in the case of Celeritas this situation won't arise, because
 * we always want to build host code in C++ files for development ease and to
 * allow testing when CUDA is disabled.)
 *
 * \todo It would be easy to specialize the traits for the const_reference
 * ownership so that for device primitive data types (int, double) we access
 * via __ldg -- speeding up everywhere in the code without any invasive
 * changes. This is another good argument for using Collection instead of Span
 * for device-compatible helper classes (e.g. grid calculator).
 */
template<class T, Ownership W, MemSpace M, class I = ItemId<T>>
class Collection
{
    using CollectionTraitsT = detail::CollectionTraits<T, W>;

  public:
    //!@{
    //! Type aliases
    using SpanT          = typename CollectionTraitsT::SpanT;
    using SpanConstT     = typename CollectionTraitsT::SpanConstT;
    using pointer        = typename CollectionTraitsT::pointer;
    using const_pointer  = typename CollectionTraitsT::const_pointer;
    using reference_type = typename CollectionTraitsT::reference_type;
    using const_reference_type =
        typename CollectionTraitsT::const_reference_type;
    using size_type            = typename I::size_type;
    using ItemIdT              = I;
    using ItemRangeT           = Range<ItemIdT>;
    //!@}

  public:
    //// CONSTRUCTION ////

    //!@{
    //! Default constructors
    Collection()                  = default;
    Collection(const Collection&) = default;
    Collection(Collection&&)      = default;
    //!@}

    // Construct from another collection
    template<Ownership W2, MemSpace M2>
    inline Collection(const Collection<T, W2, M2, I>& other);

    // Construct from another collection (mutable)
    template<Ownership W2, MemSpace M2>
    inline Collection(Collection<T, W2, M2, I>& other);

    //!@{
    //! Default assignment
    Collection& operator=(const Collection& other) = default;
    Collection& operator=(Collection&& other) = default;
    //!@}

    // Assign from another collection in the same memory space
    template<Ownership W2>
    inline Collection& operator=(const Collection<T, W2, M, I>& other);

    // Assign (mutable!) from another collection in the same memory space
    template<Ownership W2>
    inline Collection& operator=(Collection<T, W2, M, I>& other);

    //// ACCESS ////

    // Access a subset of the data with a slice
    inline CELER_FUNCTION SpanT      operator[](ItemRangeT ps);
    inline CELER_FUNCTION SpanConstT operator[](ItemRangeT ps) const;

    // Access a single element
    inline CELER_FUNCTION reference_type       operator[](ItemIdT i);
    inline CELER_FUNCTION const_reference_type operator[](ItemIdT i) const;

    // Direct accesors to underlying data
    CELER_CONSTEXPR_FUNCTION size_type     size() const;
    CELER_CONSTEXPR_FUNCTION bool          empty() const;
    CELER_CONSTEXPR_FUNCTION const_pointer data() const;
    CELER_CONSTEXPR_FUNCTION pointer       data();

  private:
    //// DATA ////

    detail::CollectionStorage<T, W, M> storage_{};

    //// FRIENDS ////

    template<class T2, Ownership W2, MemSpace M2, class Id2>
    friend class Collection;

    template<class T2, MemSpace M2, class Id2>
    friend class CollectionBuilder;

  protected:
    //!@{
    // Private accessors for collection construction/access
    using StorageT = typename detail::CollectionStorage<T, W, M>::type;
    CELER_FORCEINLINE_FUNCTION const StorageT& storage() const
    {
        return storage_.data;
    }
    CELER_FORCEINLINE_FUNCTION StorageT& storage() { return storage_.data; }
    //@}
};

//! Collection for data of type T but indexed by ThreadId for use in States
template<class T, Ownership W, MemSpace M>
using StateCollection = Collection<T, W, M, ThreadId>;

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "Collection.i.hh"
