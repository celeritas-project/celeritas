//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/CollectionAlgorithms.hh
//---------------------------------------------------------------------------//
#pragma once

#include <numeric>
#include <vector>

#include "Collection.hh"
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
 * Copy from the given collection to host.
 */
template<class T, Ownership W, MemSpace M, class I, std::size_t E>
void copy_to_host(Collection<T, W, M, I> const& src,
                  Span<T, E> dst,
                  StreamId sid = {})
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
