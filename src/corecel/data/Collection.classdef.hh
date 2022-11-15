//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/Collection.classdef.hh
//! \brief UNPLEASANT HACK TO BE INCLUDED ONLY BY Collection.hh
//---------------------------------------------------------------------------//
// There should be *no* #pragma once here because it needs to be included
// twice. Nor should there be any includes here.

#ifdef __INTELLISENSE__
// To improve formatting and error checking in IDEs, pretend that the class
// header is here. In reality, it should be in the Collection.hh wrapping this
// include.
#    define CELER_COLLECTION_FORCEINLINE
namespace celeritas
{
template<class T, Ownership W, MemSpace M, class I>
class Collection
{
#endif

#if !defined(CELER_COLLECTION_FORCEINLINE)
#    error "Macros that should be defined in Collection.hh are not present"
#endif

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

    //!@{
    /*!
     * Construct or assign from another collection.
     *
     * These are generally used to create "references" to "values" (same memory
     * space) but can also be used to copy from device to host. The \c
     * detail::CollectionAssigner class statically checks for allowable
     * transformations and memory moves.
     *
     * TODO: add optimization to do an in-place copy (rather than a new
     * allocation) if the host and destination are the same size.
     */
    template<Ownership W2, MemSpace M2>
    explicit Collection(const Collection<T, W2, M2, I>& other)
        : storage_(detail::CollectionAssigner<W, M>()(other.storage_))
    {
        detail::CollectionStorageValidator<W2>()(this->size(),
                                                 other.storage().size());
    }
    // Construct from another collection (mutable)
    template<Ownership W2, MemSpace M2>
    explicit Collection(Collection<T, W2, M2, I>& other)
        : storage_(detail::CollectionAssigner<W, M>()(other.storage_))
    {
        detail::CollectionStorageValidator<W2>()(this->size(),
                                                 other.storage().size());
    }

    // Assign from another collection
    template<Ownership W2, MemSpace M2>
    Collection& operator=(const Collection<T, W2, M2, I>& other)
    {
        storage_ = detail::CollectionAssigner<W, M>()(other.storage_);
        detail::CollectionStorageValidator<W2>()(this->size(),
                                                 other.storage().size());
        return *this;
    }
    // Assign (mutable!) from another collection
    template<Ownership W2, MemSpace M2>
    Collection& operator=(Collection<T, W2, M2, I>& other)
    {
        storage_ = detail::CollectionAssigner<W, M>()(other.storage_);
        detail::CollectionStorageValidator<W2>()(this->size(),
                                                 other.storage().size());
        return *this;
    }
    //!@}

    //---------------------------------------------------------------------------//
    //!@{
    //! Default assignment
    Collection& operator=(const Collection& other) = default;
    Collection& operator=(Collection&& other) = default;
    //!@}

    //// ACCESS ////

    //!@{
    //! Access a single element
    CELER_COLLECTION_FORCEINLINE reference_type operator[](ItemIdT i)
    {
        CELER_EXPECT(i < this->size());
        return this->storage()[i.get()];
    }
    CELER_COLLECTION_FORCEINLINE const_reference_type operator[](ItemIdT i) const
    {
        CELER_EXPECT(i < this->size());
        return this->storage()[i.get()];
    }
    //!@}

    //!@{
    //! Access a subset of the data as a Span
    CELER_COLLECTION_FORCEINLINE SpanT operator[](ItemRangeT ps)
    {
        CELER_EXPECT(*ps.end() < this->size() + 1);
        typename CollectionTraitsT::pointer data = this->storage().data();
        return {data + ps.begin()->get(), data + ps.end()->get()};
    }
    CELER_COLLECTION_FORCEINLINE SpanConstT operator[](ItemRangeT ps) const
    {
        CELER_EXPECT(*ps.end() < this->size() + 1);
        typename CollectionTraitsT::const_pointer data = this->storage().data();
        return {data + ps.begin()->get(), data + ps.end()->get()};
    }
    //!@}

    //!@{
    //! Directly access all data as a span.
    CELER_COLLECTION_FORCEINLINE SpanT operator[](AllItemsT)
    {
        return {this->storage().data(), this->storage().size()};
    }
    CELER_COLLECTION_FORCEINLINE SpanConstT operator[](AllItemsT) const
    {
        return {this->storage().data(), this->storage().size()};
    }
    //!@}

    //!@{
    //! Direct accesors to underlying data
    CELER_COLLECTION_FORCEINLINE size_type size() const
    {
        return this->storage().size();
    }
    CELER_COLLECTION_FORCEINLINE bool empty() const
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
    CELER_COLLECTION_FORCEINLINE const StorageT& storage() const
    {
        return storage_.data;
    }
    CELER_COLLECTION_FORCEINLINE StorageT& storage() { return storage_.data; }
    //@}

#ifdef __INTELLISENSE__
    // End of class wrapper
};
} // namespace celeritas
#endif
