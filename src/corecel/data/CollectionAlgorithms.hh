//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/CollectionAlgorithms.hh
//---------------------------------------------------------------------------//
#pragma once

#include <numeric>
#include <utility>
#include <vector>

#include "Collection.hh"
#include "CollectionBuilder.hh"
#include "Copier.hh"
#include "Filler.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Fill the collection with the given value.
 *
 * This should only be used during initialization when stream synchronization
 * is OK. Use the \c Filler directly during runtime since it supports streams.
 */
template<class T, Ownership W, MemSpace M, class I>
void fill(T&& value, Collection<T, W, M, I>* col)
{
    static_assert(W != Ownership::const_reference,
                  "const references cannot be filled");
    CELER_EXPECT(col);
    Filler<T, M> fill_impl{std::forward<T>(value)};
    fill_impl((*col)[AllItems<T, M>{}]);
}

//---------------------------------------------------------------------------//
/*!
 * Fill the collection with sequentially increasing values starting from zero.
 *
 * \todo Make this a kernel launch rather than a local allocation and
 * synchronous copy.
 */
template<class T, Ownership W, MemSpace M, class I>
void fill_sequence(Collection<T, W, M, I>* dst, StreamId stream)
{
    static_assert(W != Ownership::const_reference,
                  "const references cannot be filled");
    CELER_EXPECT(dst);

    std::vector<T> src(dst->size());
    std::iota(src.begin(), src.end(), T{0});
    Copier<T, M> copy{(*dst)[AllItems<T, M>{}], stream};
    copy(MemSpace::host, make_span(src));
}

//---------------------------------------------------------------------------//
/*!
 * Reallocate the collection with a larger capacity.
 *
 * The first \c num_used elements are copied into the new storage. On device,
 * the allocation, copy, and release of the old storage are all ordered on the
 * given stream, so the stream must order this after any kernels using the
 * collection.
 *
 * \warning This invalidates all references, spans, and pointers to the
 * collection's data, including copies in device memory. The caller must
 * update them before they are used again.
 */
template<class T, MemSpace M, class I>
void reserve(Collection<T, Ownership::value, M, I>* col,
             size_type num_used,
             size_type capacity,
             StreamId stream)
{
    CELER_EXPECT(col);
    CELER_EXPECT(num_used <= col->size());
    CELER_EXPECT(capacity > col->size());
    CELER_EXPECT(stream);

    Collection<T, Ownership::value, M, I> result;
    resize(&result, capacity, stream);

    if (num_used > 0)
    {
        Range<I> used{I{0}, I{num_used}};
        Copier<T, M> copy{result[used], stream};
        copy(M, (*col)[used]);
    }
    *col = std::move(result);

    CELER_ENSURE(col->size() == capacity);
}

//---------------------------------------------------------------------------//
/*!
 * Copy from the given collection to host.
 */
template<class T, Ownership W, MemSpace M, class I, std::size_t E>
void copy_to_host(
    Collection<T, W, M, I> const& src, Span<T, E> dst, StreamId sid = {})
{
    CELER_EXPECT(src.size() == dst.size());
    Copier<T, MemSpace::host> copy_to_result{dst, sid};
    copy_to_result(M, src[AllItems<T, M>{}]);
}

//---------------------------------------------------------------------------//
/*!
 * Create a new host collection from the given collection.
 *
 * This is useful for debugging.
 */
template<class T, Ownership W, MemSpace M, class I>
auto copy_to_host(Collection<T, W, M, I> const& src)
{
    Collection<T, Ownership::value, MemSpace::host, I> result;
    result = src;
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
