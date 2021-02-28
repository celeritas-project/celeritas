//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Pie.i.hh
//---------------------------------------------------------------------------//
#include "base/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from another pie.
 */
template<class T, Ownership W, MemSpace M, class I>
template<Ownership W2, MemSpace M2>
Pie<T, W, M, I>::Pie(const Pie<T, W2, M2, I>& other)
    : storage_(detail::PieAssigner<W, M>()(other.storage_))
{
    detail::PieStorageValidator<W2>()(this->size(), other.storage().size());
}

//---------------------------------------------------------------------------//
/*!
 * Construct from another pie (mutable).
 */
template<class T, Ownership W, MemSpace M, class I>
template<Ownership W2, MemSpace M2>
Pie<T, W, M, I>::Pie(Pie<T, W2, M2, I>& other)
    : storage_(detail::PieAssigner<W, M>()(other.storage_))
{
    detail::PieStorageValidator<W2>()(this->size(), other.storage().size());
}

//---------------------------------------------------------------------------//
/*!
 * Assign from another pie in the same memory space.
 */
template<class T, Ownership W, MemSpace M, class I>
template<Ownership W2>
Pie<T, W, M, I>& Pie<T, W, M, I>::operator=(const Pie<T, W2, M, I>& other)
{
    storage_ = detail::PieAssigner<W, M>()(other.storage_);
    detail::PieStorageValidator<W2>()(this->size(), other.storage().size());
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Assign (mutable!) from another pie in the same memory space.
 */
template<class T, Ownership W, MemSpace M, class I>
template<Ownership W2>
Pie<T, W, M, I>& Pie<T, W, M, I>::operator=(Pie<T, W2, M, I>& other)
{
    storage_ = detail::PieAssigner<W, M>()(other.storage_);
    detail::PieStorageValidator<W2>()(this->size(), other.storage().size());
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Access a subspan.
 */
template<class T, Ownership W, MemSpace M, class I>
CELER_FUNCTION auto Pie<T, W, M, I>::operator[](ItemRangeT ps) -> SpanT
{
    CELER_EXPECT(*ps.end() < this->size() + 1);
    return {this->data() + ps.begin()->get(), this->data() + ps.end()->get()};
}

//---------------------------------------------------------------------------//
/*!
 * Access a subspan (const).
 */
template<class T, Ownership W, MemSpace M, class I>
CELER_FUNCTION auto Pie<T, W, M, I>::operator[](ItemRangeT ps) const
    -> SpanConstT
{
    CELER_EXPECT(*ps.end() < this->size() + 1);
    return {this->data() + ps.begin()->get(), this->data() + ps.end()->get()};
}

//---------------------------------------------------------------------------//
/*!
 * Access a single element.
 */
template<class T, Ownership W, MemSpace M, class I>
CELER_FUNCTION auto Pie<T, W, M, I>::operator[](ItemIdT i) -> reference_type
{
    CELER_EXPECT(i < this->size());
    return this->storage()[i.get()];
}

//---------------------------------------------------------------------------//
/*!
 * Access a single element (const).
 */
template<class T, Ownership W, MemSpace M, class I>
CELER_FUNCTION auto Pie<T, W, M, I>::operator[](ItemIdT i) const
    -> const_reference_type
{
    CELER_EXPECT(i < this->size());
    return this->storage()[i.get()];
}

//---------------------------------------------------------------------------//
//!@{
//! Direct accesors to underlying data
template<class T, Ownership W, MemSpace M, class I>
CELER_CONSTEXPR_FUNCTION auto Pie<T, W, M, I>::size() const -> size_type
{
    return this->storage().size();
}

template<class T, Ownership W, MemSpace M, class I>
CELER_CONSTEXPR_FUNCTION bool Pie<T, W, M, I>::empty() const
{
    return this->storage().empty();
}

template<class T, Ownership W, MemSpace M, class I>
CELER_CONSTEXPR_FUNCTION auto Pie<T, W, M, I>::data() const -> const_pointer
{
    return this->storage().data();
}

template<class T, Ownership W, MemSpace M, class I>
CELER_CONSTEXPR_FUNCTION auto Pie<T, W, M, I>::data() -> pointer
{
    return this->storage().data();
}
//!@}

//---------------------------------------------------------------------------//
} // namespace celeritas
