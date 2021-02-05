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
 * Construct with a particular range.
 */
template<class T>
CELER_FUNCTION PieSlice<T>::PieSlice(size_type start, size_type stop)
    : start_(start), stop_(stop)
{
    CELER_EXPECT(start_ <= stop_);
}

//---------------------------------------------------------------------------//
/*!
 * Construct from another pie.
 */
template<class T, Ownership W, MemSpace M>
template<Ownership W2, MemSpace M2>
Pie<T, W, M>::Pie(const Pie<T, W2, M2>& other)
    : storage_(detail::PieAssigner<W, M>()(other.storage_))
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct from another pie (mutable).
 */
template<class T, Ownership W, MemSpace M>
template<Ownership W2, MemSpace M2>
Pie<T, W, M>::Pie(Pie<T, W2, M2>& other)
    : storage_(detail::PieAssigner<W, M>()(other.storage_))
{
}

//---------------------------------------------------------------------------//
/*!
 * Assign from another pie in the same memory space.
 */
template<class T, Ownership W, MemSpace M>
template<Ownership W2>
Pie<T, W, M>& Pie<T, W, M>::operator=(const Pie<T, W2, M>& other)
{
    storage_ = detail::PieAssigner<W, M>()(other.storage_);
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Assign (mutable!) from another pie in the same memory space.
 */
template<class T, Ownership W, MemSpace M>
template<Ownership W2>
Pie<T, W, M>& Pie<T, W, M>::operator=(Pie<T, W2, M>& other)
{
    storage_ = detail::PieAssigner<W, M>()(other.storage_);
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Access a subspan.
 */
template<class T, Ownership W, MemSpace M>
CELER_FUNCTION auto Pie<T, W, M>::operator[](const PieSlice<T>& ps) -> SpanT
{
    CELER_EXPECT(ps.stop() <= this->size());
    return {this->data() + ps.start(), this->data() + ps.stop()};
}

//---------------------------------------------------------------------------//
/*!
 * Access a subspan (const).
 */
template<class T, Ownership W, MemSpace M>
CELER_FUNCTION auto Pie<T, W, M>::operator[](const PieSlice<T>& ps) const
    -> SpanConstT
{
    CELER_EXPECT(ps.stop() <= this->size());
    return {this->data() + ps.start(), this->data() + ps.stop()};
}

//---------------------------------------------------------------------------//
/*!
 * Access a single element.
 */
template<class T, Ownership W, MemSpace M>
CELER_FUNCTION auto Pie<T, W, M>::operator[](size_type i) -> reference_type
{
    CELER_EXPECT(i < this->size());
    return this->storage()[i];
}

//---------------------------------------------------------------------------//
/*!
 * Access a single element (const).
 */
template<class T, Ownership W, MemSpace M>
CELER_FUNCTION auto Pie<T, W, M>::operator[](size_type i) const
    -> const_reference_type
{
    CELER_EXPECT(i < this->size());
    return this->storage()[i];
}

//---------------------------------------------------------------------------//
//!@{
//! Direct accesors to underlying data
template<class T, Ownership W, MemSpace M>
CELER_CONSTEXPR_FUNCTION auto Pie<T, W, M>::size() const -> size_type
{
    return this->storage().size();
}

template<class T, Ownership W, MemSpace M>
CELER_CONSTEXPR_FUNCTION bool Pie<T, W, M>::empty() const
{
    return this->storage().empty();
}

template<class T, Ownership W, MemSpace M>
CELER_CONSTEXPR_FUNCTION auto Pie<T, W, M>::data() const -> const_pointer
{
    return this->storage().data();
}

template<class T, Ownership W, MemSpace M>
CELER_CONSTEXPR_FUNCTION auto Pie<T, W, M>::data() -> pointer
{
    return this->storage().data();
}
//!@}
//---------------------------------------------------------------------------//
} // namespace celeritas
