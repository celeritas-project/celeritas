//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/CollectionAlgorithms.hh
//---------------------------------------------------------------------------//
#pragma once

#include "Collection.hh"
#include "Copier.hh"
#include "detail/Filler.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Fill the collection with the given value.
 */
template<class T, Ownership W, MemSpace M, class I>
void fill(T&& value, Collection<T, W, M, I>* col)
{
    static_assert(W != Ownership::const_reference,
                  "const references cannot be filled");
    CELER_EXPECT(col);
    detail::Filler<T, M> fill_impl{value};
    fill_impl((*col)[AllItems<T, M>{}]);
}

//---------------------------------------------------------------------------//
/*!
 * Copy from the given collection to host.
 */
template<class T, Ownership W, MemSpace M, class I, std::size_t E>
void copy_to_host(Collection<T, W, M, I> const& src, Span<T, E> dst)
{
    CELER_EXPECT(src.size() == dst.size());
    Copier<T, MemSpace::host> copy_to_result{dst};
    copy_to_result(M, src[AllItems<T, M>{}]);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
