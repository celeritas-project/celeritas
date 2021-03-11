//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file CollectionAlgorithms.hh
//---------------------------------------------------------------------------//
#pragma once

#include "Collection.hh"
#include "detail/Copier.hh"
#include "detail/Filler.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Fill the collection with the given value.
 */
template<class T, MemSpace M, class I>
void fill(const T& value, Collection<T, Ownership::value, M, I>* col)
{
    CELER_EXPECT(col);
    detail::Filler<T, M> fill_impl{value};
    fill_impl((*col)[AllItems<T, M>{}]);
}

//---------------------------------------------------------------------------//
/*!
 * Copy from the given collection to host.
 */
template<class T, Ownership W, MemSpace M, class I, std::size_t E>
void copy_to_host(const Collection<T, W, M, I>& src, Span<T, E> dst)
{
    CELER_EXPECT(src.size() == dst.size());
    detail::Copier<T, M> copy_impl{src[AllItems<T, M>{}]};
    copy_impl(MemSpace::host, dst);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
