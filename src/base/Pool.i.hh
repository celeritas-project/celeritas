//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Pool.i.hh
//---------------------------------------------------------------------------//

#include "base/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with a particular range.
 */
template<class T>
CELER_FUNCTION PoolSlice<T>::PoolSlice(size_type start, size_type stop)
    : start_(start), stop_(stop)
{
    CELER_EXPECT(start_ <= stop_);
}

//---------------------------------------------------------------------------//
/*!
 * Construct from another pool.
 */
template<class T, Ownership W, MemSpace M>
template<Ownership W2, MemSpace M2>
Pool<T, W, M>::Pool(const Pool<T, W2, M2>& other)
    : storage_(detail::PoolAssigner<W, M>()(other.storage_))
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct from another pool (mutable).
 */
template<class T, Ownership W, MemSpace M>
template<Ownership W2, MemSpace M2>
Pool<T, W, M>::Pool(Pool<T, W2, M2>& other)
    : storage_(detail::PoolAssigner<W, M>()(other.storage_))
{
}

//---------------------------------------------------------------------------//
/*!
 * Assign from another pool in the same memory space.
 */
template<class T, Ownership W, MemSpace M>
template<Ownership W2>
Pool<T, W, M>& Pool<T, W, M>::operator=(const Pool<T, W2, M>& other)
{
    storage_ = detail::PoolAssigner<W, M>()(other.storage_);
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Assign (mutable!) from another pool in the same memory space.
 */
template<class T, Ownership W, MemSpace M>
template<Ownership W2>
Pool<T, W, M>& Pool<T, W, M>::operator=(Pool<T, W2, M>& other)
{
    storage_ = detail::PoolAssigner<W, M>()(other.storage_);
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Access a subspan.
 */
template<class T, Ownership W, MemSpace M>
CELER_FUNCTION auto Pool<T, W, M>::operator[](const PoolSlice<T>& ps) const
    -> SpanT
{
    CELER_EXPECT(ps.stop() <= this->size());
    return {this->data() + ps.start(), this->data() + ps.stop()};
}

//---------------------------------------------------------------------------//
/*!
 * Access a single element.
 */
template<class T, Ownership W, MemSpace M>
CELER_FUNCTION auto Pool<T, W, M>::operator[](size_type i) const
    -> reference_type
{
    CELER_EXPECT(i < this->size());
    return this->storage()[i];
}

//---------------------------------------------------------------------------//
//!@{
//! Direct accesors to underlying data
template<class T, Ownership W, MemSpace M>
CELER_CONSTEXPR_FUNCTION auto Pool<T, W, M>::size() const -> size_type
{
    return this->storage().size();
}

template<class T, Ownership W, MemSpace M>
CELER_CONSTEXPR_FUNCTION bool Pool<T, W, M>::empty() const
{
    return this->storage().empty();
}

template<class T, Ownership W, MemSpace M>
CELER_CONSTEXPR_FUNCTION auto Pool<T, W, M>::data() const -> const_pointer
{
    return this->storage().data();
}

template<class T, Ownership W, MemSpace M>
CELER_CONSTEXPR_FUNCTION auto Pool<T, W, M>::data() -> pointer
{
    return this->storage().data();
}
//!@}
//---------------------------------------------------------------------------//
} // namespace celeritas
