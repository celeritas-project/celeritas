//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Collection.hh
//---------------------------------------------------------------------------//
#pragma once

#include "OpaqueId.hh"
#include "Range.hh"
#include "Types.hh"
#include "detail/CollectionImpl.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * \page collections Collection: a data portability class
 *
 * The \c Collection manages data allocation and transfer between CPU and GPU.
 * Its primary design goal is facilitating construction of deeply hierarchical
 * data on host at setup time and seamlessly copying to device.
 * The templated \c T must be trivially copyable---either a fundamental data
 * type or a struct of such types.
 *
 * An individual item in a \c Collection<T> can be accessed with \c ItemId<T>,
 * a contiguous subset of items are accessed with \c ItemRange<T>, and the
 * entirety of the data are accessed with \c AllItems<T>. All three of these
 * classes are trivially copyable, so they can be embedded in structs that can
 * be managed by a Collection. A group of Collections, one for each data type,
 * can therefore be trivially copied to the GPU to enable arbitrarily deep and
 * complex data hierarchies.
 *
 * By convention, groups of Collections comprising the data for a single class
 * or subsystem (such as RayleighInteractor or Physics) are stored in a helper
 * struct suffixed with \c Data . For cases where there is both persistent data
 * (problem-specific parameters) and transient data (track-specific states),
 * the collections must be grouped into two separate classes. \c StateData are
 * meant to be mutable and never directly copied between host and device; its
 * data collections are typically accessed by thread ID. \c ParamsData are
 * immutable and always "mirrored" on both host and device. Sometimes it's
 * sensible to partition \c ParamsData into discrete helper structs (stored by
 * value), each with a group of collections, and perhaps another struct that
 * has non-templated scalars (since the default assignment operator is less
 * work than manually copying scalars in a templated assignment operator.
 *
 * A collection group has the following requirements to be compatible with the
\c
 * CollectionMirror, \c CollectionStateStore, and other such helper classes:
 * - Be a struct templated with \c template<Ownership W, MemSpace M>
 * - Contain only Collection objects and trivially copyable structs
 * - Define an operator bool that is true if and only if the class data is
 *   assigned and consistent
 * - Define a templated assignment operator on "other" Ownership and MemSpace
 *   which assigns every member to the right-hand-side's member
 *
 * Additionally, a \c StateData collection group must define
 * - A member function \c size() returning the number of entries (i.e. number
 *   of threads)
 * - A free function \c resize with one of two signatures:
 * \code
void resize(
    StateData<Ownership::value, M>*                               data,
    const ParamsData<Ownership::const_reference, MemSpace::host>& params,
    size_type                                                     size);
// or...
void resize(
    StateData<Ownership::value, M>* data,
    size_type                       size);
 * \endcode
 *
 * By convention, related groups of collections are stored in a header file
 * named \c Data.hh .
 *
 * See ParticleParamsData and ParticleStateData for minimal examples of using
 * collections. The MaterialParamsData demonstrates additional complexity
 * by having a multi-level data hierarchy, and MaterialStateData has a resize
 * function that uses params data. PhysicsParamsData is a very complex example,
 * and GeoParamsData demonstates how to use template specialization to adapt
 * Collections to another codebase with a different convention for host-device
 * portability.
 */

//! Opaque ID representing a single element of a container.
template<class T>
using ItemId = OpaqueId<T, size_type>;

//---------------------------------------------------------------------------//
/*!
 * Reference a contiguous range of IDs corresponding to a slice of items.
 *
 * \tparam T The value type of items to represent.
 *
 * An ItemRange is a range of \c OpaqueId<T> that reference a range of values
 * of type \c T in a \c Collection . The ItemRange acts like a \c slice object
 * in Python when used on a Collection, returning a Span<T> of the underlying
 * data.
 *
 * An ItemRange is only meaningful in connection with a particular Collection
 * of type T. It doesn't have any persistent connection to its associated
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
template<class T, class Size = size_type>
using ItemRange = Range<OpaqueId<T, Size>>;

// Forward-declare collection builder, needed for GCC7
template<class T2, MemSpace M2, class Id2>
class CollectionBuilder;

//---------------------------------------------------------------------------//
/*!
 * Sentinel class for obtaining a view to all items of a collection.
 */
template<class T, MemSpace M = MemSpace::native>
struct AllItems
{
};

//---------------------------------------------------------------------------//
/*!
 * Manage generic array-like data ownership and transfer from host to device.
 *
 * Data are constructed incrementally on the host, then copied (along with
 * their associated ItemRange) to device. A Collection can act as a
 * std::vector<T>, DeviceVector<T>, Span<T>, or Span<const T>. The Spans can
 * point to host or device memory, but the MemSpace template argument protects
 * against accidental accesses from the wrong memory space.
 *
 * Each Collection object is usually accessed with an ItemRange, which
 * references a
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
    using reference_type = typename CollectionTraitsT::reference_type;
    using const_reference_type =
        typename CollectionTraitsT::const_reference_type;
    using size_type  = typename I::size_type;
    using value_type = T;
    using ItemIdT    = I;
    using ItemRangeT = Range<ItemIdT>;
    using AllItemsT  = AllItems<T, M>;
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
    explicit inline Collection(const Collection<T, W2, M2, I>& other);

    // Construct from another collection (mutable)
    template<Ownership W2, MemSpace M2>
    explicit inline Collection(Collection<T, W2, M2, I>& other);

    //!@{
    //! Default assignment
    Collection& operator=(const Collection& other) = default;
    Collection& operator=(Collection&& other) = default;
    //!@}

    // Assign from another collection
    template<Ownership W2, MemSpace M2>
    inline Collection& operator=(const Collection<T, W2, M2, I>& other);

    // Assign (mutable!) from another collection
    template<Ownership W2, MemSpace M2>
    inline Collection& operator=(Collection<T, W2, M2, I>& other);

    //// ACCESS ////

    // Access a single element
    inline CELER_FUNCTION reference_type       operator[](ItemIdT i);
    inline CELER_FUNCTION const_reference_type operator[](ItemIdT i) const;

    // Access a subset of the data with a slice
    inline CELER_FUNCTION SpanT      operator[](ItemRangeT ps);
    inline CELER_FUNCTION SpanConstT operator[](ItemRangeT ps) const;

    // Access all data.
    inline CELER_FUNCTION SpanT      operator[](AllItemsT);
    inline CELER_FUNCTION SpanConstT operator[](AllItemsT) const;

    //!@{
    //! Direct accesors to underlying data
    CELER_FORCEINLINE_FUNCTION size_type size() const
    {
        return this->storage().size();
    }
    CELER_FORCEINLINE_FUNCTION bool empty() const
    {
        return this->storage().empty();
    }
    //!@}

  private:
    //// DATA ////

    detail::CollectionStorage<T, W, M> storage_{};

  protected:
    //// FRIENDS ////

    template<class T2, Ownership W2, MemSpace M2, class Id2>
    friend class Collection;

    template<class T2, MemSpace M2, class Id2>
    friend class CollectionBuilder;

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
