//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PieBuilder.i.hh
//---------------------------------------------------------------------------//

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Reserve space for the given number of elements.
 */
template<class T, MemSpace M, class I>
void PieBuilder<T, M, I>::reserve(size_type count)
{
    CELER_EXPECT(count <= max_pie_size());
    this->storage().reserve(count);
}

//---------------------------------------------------------------------------//
/*!
 * Insert the given elements at the end of the allocation.
 */
template<class T, MemSpace M, class I>
template<class InputIterator>
auto PieBuilder<T, M, I>::insert_back(InputIterator first, InputIterator last)
    -> PieSliceT
{
    CELER_EXPECT(std::distance(first, last) + this->storage().size()
                 <= this->max_pie_size());
    static_assert(M == MemSpace::host,
                  "Insertion currently works only for host memory");
    auto start = PieIndexT{this->size()};
    this->storage().insert(this->storage().end(), first, last);
    return {start, PieIndexT{this->size()}};
}

//---------------------------------------------------------------------------//
/*!
 * Insert the given list of elements at the end of the allocation.
 */
template<class T, MemSpace M, class I>
auto PieBuilder<T, M, I>::insert_back(std::initializer_list<T> init)
    -> PieSliceT
{
    return this->insert_back(init.begin(), init.end());
}

//---------------------------------------------------------------------------//
/*!
 * Reserve space for the given number of elements.
 */
template<class T, MemSpace M, class I>
auto PieBuilder<T, M, I>::push_back(T el) -> PieIndexT
{
    CELER_EXPECT(this->storage().size() + 1 <= this->max_pie_size());
    static_assert(M == MemSpace::host,
                  "Insertion currently works only for host memory");
    size_type idx = this->size();
    this->storage().push_back(el);
    return PieIndexT{idx};
}

//---------------------------------------------------------------------------//
// DEVICE PIE BUILDDER
//---------------------------------------------------------------------------//
/*!
 * Increase the size to the given number of elements.
 *
 * \todo Rethink whether to add resizing to DeviceVector, since this
 * construction is super awkward.
 */
template<class T, MemSpace M, class I>
void PieBuilder<T, M, I>::resize(size_type size)
{
    CELER_EXPECT(size >= this->size());
    CELER_EXPECT(this->storage().empty() || size <= this->storage().capacity());
    if (this->storage().empty())
    {
        this->storage() = StorageT(size);
    }
    else
    {
        this->storage().resize(size);
    }
}

//---------------------------------------------------------------------------//
} // namespace celeritas
