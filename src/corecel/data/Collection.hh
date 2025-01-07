//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/Collection.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/OpaqueId.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/sys/ThreadId.hh"

#include "ObserverPtr.hh"

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
       StateData<Ownership::value, M>* data,
       HostCRef<ParamsData> const&     params,
       StreamId                        stream,
       size_type                       size);
   // or...
   void resize(
       StateData<Ownership::value, M>* data,
       const HostCRef<ParamsData>&     params,
       size_type                       size);
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
 * \code
   struct MyMaterial
   {
       real_type number_density;
       ItemRange<ElementComponents> components;
   };

   template<Ownership W, MemSpace M>
   struct MyData
   {
       Collection<ElementComponents, W, M> components;
       Collection<MyMaterial, W, M> materials;
   };
 * \endcode
 */
template<class T, class Size = size_type>
using ItemRange = Range<OpaqueId<T, Size>>;

//---------------------------------------------------------------------------//
/*!
 * Access data in a Range<T2> with an index of type T1.
 *
 * Here, T1 and T2 are expected to be OpaqueId types. This is simply a
 * type-safe "offset" with range checking.
 */
template<class T1, class T2>
class ItemMap
{
    static_assert(detail::is_opaque_id_v<T1>, "T1 is not OpaqueID");
    static_assert(detail::is_opaque_id_v<T2>, "T2 is not OpaqueID");

  public:
    //!@{
    //! \name Type aliases
    using key_type = T1;
    using mapped_type = T2;
    //!@}

  public:
    //// CONSTRUCTION ////

    ItemMap() = default;

    //! Contruct from an exising Range<T2>
    explicit CELER_FUNCTION ItemMap(Range<T2> range) : range_(range) {}

    //// ACCESS ////

    //! Access Range via OpaqueId of type T1
    CELER_FORCEINLINE_FUNCTION T2 operator[](T1 id) const
    {
        CELER_EXPECT(id < this->size());
        return range_[id.unchecked_get()];
    }

    //! Whether the underlying Range<T2> is empty
    CELER_FORCEINLINE_FUNCTION bool empty() const { return range_.empty(); }

    //! Size of the underlying Range<T2>
    CELER_FORCEINLINE_FUNCTION size_type size() const { return range_.size(); }

  private:
    //// DATA ////
    Range<T2> range_;
};

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
 * A \c MemSpace::Mapped collection will be accessible on the host and the
 * device. Unified addressing must be supported by the current device or an
 * exception will be thrown when using the collection. Mapped pinned memory
 * (i.e. zero-copy memory) is allocated, pages will always reside on host
 * memory and each access from device code will require a slow memory transfer.
 * Allocating pinned memory is slow and reduce the memory available to the
 * system: only allocate the smallest amount needed with the longest possible
 * lifetime. Frequently accessing data from device code will result in low
 * performance. Usecase for this MemSapce are: as a src / dst memory space for
 * asynchronous operations, on integrated GPU architecture, or a single
 * coalesced read or write from device code.
 * https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#zero-copy
 *
 * Accessing a \c const_reference collection in \c device memory will return a
 * wrapper container that accesses the low-level data through the \c __ldg
 * primitive, which can accelerate random access by telling the compiler
 * <em>the memory will not be changed during the lifetime of the kernel</em>.
 * Therefore it is important to \em only use Collections for shared,
 * constant "params" data.
 */
template<class T, Ownership W, MemSpace M, class I = ItemId<T>>
class Collection
{
    // rocrand states have nontrivial destructors
    static_assert(std::is_trivially_copyable<T>::value || CELERITAS_USE_HIP,
                  "Collection element is not trivially copyable");
    static_assert(std::is_trivially_destructible<T>::value || CELERITAS_USE_HIP,
                  "Collection element is not trivially destructible");

    using CollectionTraitsT = detail::CollectionTraits<T, W, M>;
    using const_value_type = typename CollectionTraitsT::const_type;

  public:
    //!@{
    //! \name Type aliases
    using value_type = typename CollectionTraitsT::type;
    using SpanT = typename CollectionTraitsT::SpanT;
    using SpanConstT = typename CollectionTraitsT::SpanConstT;
    using pointer = ObserverPtr<value_type, M>;
    using const_pointer = ObserverPtr<const_value_type, M>;
    using reference_type = typename CollectionTraitsT::reference_type;
    using const_reference_type =
        typename CollectionTraitsT::const_reference_type;
    using size_type = typename I::size_type;
    using ItemIdT = I;
    using ItemRangeT = Range<ItemIdT>;
    using AllItemsT = AllItems<T, M>;
    //!@}

    static constexpr inline Ownership ownership = W;
    static constexpr inline MemSpace memspace = M;

  public:
    //// CONSTRUCTION ////

    //!@{
    //! Default constructors
    Collection() = default;
    Collection(Collection const&) = default;
    Collection(Collection&&) = default;
    //!@}

    ~Collection() = default;

    // Construct from another collection
    template<Ownership W2, MemSpace M2>
    explicit inline Collection(Collection<T, W2, M2, I> const& other);

    // Construct from another collection (mutable)
    template<Ownership W2, MemSpace M2>
    explicit inline Collection(Collection<T, W2, M2, I>& other);

    //!@{
    //! Default assignment
    Collection& operator=(Collection const& other) = default;
    Collection& operator=(Collection&& other) = default;
    //!@}

    // Assign from another collection
    template<Ownership W2, MemSpace M2>
    inline Collection& operator=(Collection<T, W2, M2, I> const& other);

    // Assign (mutable!) from another collection
    template<Ownership W2, MemSpace M2>
    inline Collection& operator=(Collection<T, W2, M2, I>& other);

    //// ACCESS ////

    // Access a single element
    CELER_FORCEINLINE_FUNCTION reference_type operator[](ItemIdT i);
    CELER_FORCEINLINE_FUNCTION const_reference_type operator[](ItemIdT i) const;

    // Access a subset of the data with a slice
    CELER_FORCEINLINE_FUNCTION SpanT operator[](ItemRangeT ps);
    CELER_FORCEINLINE_FUNCTION SpanConstT operator[](ItemRangeT ps) const;

    // Access all data.
    CELER_FORCEINLINE_FUNCTION SpanT operator[](AllItemsT);
    CELER_FORCEINLINE_FUNCTION SpanConstT operator[](AllItemsT) const;

    //!@{
    //! Direct accesors to underlying data
    CELER_FORCEINLINE_FUNCTION size_type size() const
    {
        return static_cast<size_type>(this->storage().size());
    }
    CELER_FORCEINLINE_FUNCTION bool empty() const
    {
        return this->storage().empty();
    }
    CELER_FORCEINLINE_FUNCTION pointer data()
    {
        return pointer{this->storage().data()};
    }
    CELER_FORCEINLINE_FUNCTION const_pointer data() const
    {
        return const_pointer{this->storage().data()};
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

    template<class T2, class Id2>
    friend class DedupeCollectionBuilder;

    //!@{
    // Private accessors for collection construction/access
    using StorageT = typename detail::CollectionStorage<T, W, M>::type;
    CELER_FORCEINLINE_FUNCTION StorageT const& storage() const
    {
        return storage_.data;
    }
    CELER_FORCEINLINE_FUNCTION StorageT& storage() { return storage_.data; }
    //@}
};

//! Collection for data of type T but indexed by TrackSlotId for use in States
template<class T, Ownership W, MemSpace M>
using StateCollection = Collection<T, W, M, TrackSlotId>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
//!@{
/*!
 * Construct or assign from another collection.
 *
 * These are generally used to create "references" to "values" (same memory
 * space) but can also be used to copy from device to host. The \c
 * detail::CollectionAssigner class statically checks for allowable
 * transformations and memory moves.
 *
 * TODO: add optimization to do an in-place copy (rather than a new allocation)
 * if the host and destination are the same size.
 */
template<class T, Ownership W, MemSpace M, class I>
template<Ownership W2, MemSpace M2>
Collection<T, W, M, I>::Collection(Collection<T, W2, M2, I> const& other)
{
    detail::copy_collection(other.storage_, &storage_);
    detail::CollectionStorageValidator<W2>()(this->size(),
                                             other.storage().size());
}

template<class T, Ownership W, MemSpace M, class I>
template<Ownership W2, MemSpace M2>
Collection<T, W, M, I>::Collection(Collection<T, W2, M2, I>& other)
{
    detail::copy_collection(other.storage_, &storage_);
    detail::CollectionStorageValidator<W2>()(this->size(),
                                             other.storage().size());
}

template<class T, Ownership W, MemSpace M, class I>
template<Ownership W2, MemSpace M2>
Collection<T, W, M, I>&
Collection<T, W, M, I>::operator=(Collection<T, W2, M2, I> const& other)
{
    detail::copy_collection(other.storage_, &storage_);
    detail::CollectionStorageValidator<W2>()(this->size(),
                                             other.storage().size());
    return *this;
}

template<class T, Ownership W, MemSpace M, class I>
template<Ownership W2, MemSpace M2>
Collection<T, W, M, I>&
Collection<T, W, M, I>::operator=(Collection<T, W2, M2, I>& other)
{
    detail::copy_collection(other.storage_, &storage_);
    detail::CollectionStorageValidator<W2>()(this->size(),
                                             other.storage().size());
    return *this;
}
//!@}

//---------------------------------------------------------------------------//
/*!
 * Access a single element.
 */
template<class T, Ownership W, MemSpace M, class I>
CELER_FUNCTION auto
Collection<T, W, M, I>::operator[](ItemIdT i) -> reference_type
{
    CELER_EXPECT(i < this->size());
    return this->storage()[i.unchecked_get()];
}

//---------------------------------------------------------------------------//
/*!
 * Access a single element (const).
 */
template<class T, Ownership W, MemSpace M, class I>
CELER_FUNCTION auto
Collection<T, W, M, I>::operator[](ItemIdT i) const -> const_reference_type
{
    CELER_EXPECT(i < this->size());
    return this->storage()[i.unchecked_get()];
}

//---------------------------------------------------------------------------//
/*!
 * Access a subset of the data as a Span.
 */
template<class T, Ownership W, MemSpace M, class I>
CELER_FUNCTION auto Collection<T, W, M, I>::operator[](ItemRangeT ps) -> SpanT
{
    CELER_EXPECT(*ps.begin() <= *ps.end());
    CELER_EXPECT(*ps.end() < this->size() + 1);
    auto* data = this->storage().data();
    return {data + ps.begin()->unchecked_get(),
            data + ps.end()->unchecked_get()};
}

//---------------------------------------------------------------------------//
/*!
 * Access a subset of the data as a Span (const).
 */
template<class T, Ownership W, MemSpace M, class I>
CELER_FUNCTION auto
Collection<T, W, M, I>::operator[](ItemRangeT ps) const -> SpanConstT
{
    CELER_EXPECT(*ps.begin() <= *ps.end());
    CELER_EXPECT(*ps.end() < this->size() + 1);
    auto* data = this->storage().data();
    return {data + ps.begin()->unchecked_get(),
            data + ps.end()->unchecked_get()};
}

//---------------------------------------------------------------------------//
/*!
 * Access all of the data as a Span.
 */
template<class T, Ownership W, MemSpace M, class I>
CELER_FUNCTION auto Collection<T, W, M, I>::operator[](AllItemsT) -> SpanT
{
    return {this->storage().data(), this->storage().size()};
}

//---------------------------------------------------------------------------//
/*!
 * Access all of the data as a Span (const).
 */
template<class T, Ownership W, MemSpace M, class I>
CELER_FUNCTION auto
Collection<T, W, M, I>::operator[](AllItemsT) const -> SpanConstT
{
    return {this->storage().data(), this->storage().size()};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
