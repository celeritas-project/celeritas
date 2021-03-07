//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Collection.i.hh
//---------------------------------------------------------------------------//
#include "base/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//!@{
/*!
 * Construct or assign from another collection.
 *
 * These are generally used to create "references" to "values" (same memory
 * space) but can also be used to copy from device to host. The \c
 * detail::CollectionAssigner class statically checks for allowable
 * transformations and memory moves.
 */
template<class T, Ownership W, MemSpace M, class I>
template<Ownership W2, MemSpace M2>
Collection<T, W, M, I>::Collection(const Collection<T, W2, M2, I>& other)
    : storage_(detail::CollectionAssigner<W, M>()(other.storage_))
{
    detail::CollectionStorageValidator<W2>()(this->size(),
                                             other.storage().size());
}

template<class T, Ownership W, MemSpace M, class I>
template<Ownership W2, MemSpace M2>
Collection<T, W, M, I>::Collection(Collection<T, W2, M2, I>& other)
    : storage_(detail::CollectionAssigner<W, M>()(other.storage_))
{
    detail::CollectionStorageValidator<W2>()(this->size(),
                                             other.storage().size());
}

template<class T, Ownership W, MemSpace M, class I>
template<Ownership W2, MemSpace M2>
Collection<T, W, M, I>&
Collection<T, W, M, I>::operator=(const Collection<T, W2, M2, I>& other)
{
    storage_ = detail::CollectionAssigner<W, M>()(other.storage_);
    detail::CollectionStorageValidator<W2>()(this->size(),
                                             other.storage().size());
    return *this;
}

template<class T, Ownership W, MemSpace M, class I>
template<Ownership W2, MemSpace M2>
Collection<T, W, M, I>&
Collection<T, W, M, I>::operator=(Collection<T, W2, M2, I>& other)
{
    storage_ = detail::CollectionAssigner<W, M>()(other.storage_);
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
CELER_FUNCTION auto Collection<T, W, M, I>::operator[](ItemIdT i)
    -> reference_type
{
    CELER_EXPECT(i < this->size());
    return this->storage()[i.get()];
}

//---------------------------------------------------------------------------//
/*!
 * Access a single element (const).
 */
template<class T, Ownership W, MemSpace M, class I>
CELER_FUNCTION auto Collection<T, W, M, I>::operator[](ItemIdT i) const
    -> const_reference_type
{
    CELER_EXPECT(i < this->size());
    return this->storage()[i.get()];
}

//---------------------------------------------------------------------------//
/*!
 * Access a subset of the data as a Span.
 */
template<class T, Ownership W, MemSpace M, class I>
CELER_FUNCTION auto Collection<T, W, M, I>::operator[](ItemRangeT ps) -> SpanT
{
    CELER_EXPECT(*ps.end() < this->size() + 1);
    typename CollectionTraitsT::pointer data = this->storage().data();
    return {data + ps.begin()->get(), data + ps.end()->get()};
}

//---------------------------------------------------------------------------//
/*!
 * Access a subset of the data as a Span (const).
 */
template<class T, Ownership W, MemSpace M, class I>
CELER_FUNCTION auto Collection<T, W, M, I>::operator[](ItemRangeT ps) const
    -> SpanConstT
{
    CELER_EXPECT(*ps.end() < this->size() + 1);
    typename CollectionTraitsT::const_pointer data = this->storage().data();
    return {data + ps.begin()->get(), data + ps.end()->get()};
}

//---------------------------------------------------------------------------//
/*!
 * Access a subset of the data as a Span.
 */
template<class T, Ownership W, MemSpace M, class I>
CELER_FUNCTION auto Collection<T, W, M, I>::operator[](AllItemsT) -> SpanT
{
    return {this->storage().data(), this->storage().size()};
}

//---------------------------------------------------------------------------//
/*!
 * Access a subset of the data as a Span (const).
 */
template<class T, Ownership W, MemSpace M, class I>
CELER_FUNCTION auto Collection<T, W, M, I>::operator[](AllItemsT) const
    -> SpanConstT
{
    return {this->storage().data(), this->storage().size()};
}

//---------------------------------------------------------------------------//
//!@{
//! Direct accesors to underlying data
template<class T, Ownership W, MemSpace M, class I>
CELER_CONSTEXPR_FUNCTION auto Collection<T, W, M, I>::size() const -> size_type
{
    return this->storage().size();
}

template<class T, Ownership W, MemSpace M, class I>
CELER_CONSTEXPR_FUNCTION bool Collection<T, W, M, I>::empty() const
{
    return this->storage().empty();
}
//!@}

//---------------------------------------------------------------------------//
} // namespace celeritas
