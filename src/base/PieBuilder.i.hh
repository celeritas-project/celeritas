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
template<class T, MemSpace M>
void PieBuilder<T, M>::reserve(size_type count)
{
    CELER_EXPECT(count <= max_pie_size());
    this->storage().reserve(count);
}

//---------------------------------------------------------------------------//
/*!
 * Insert the given elements at the end of the allocation.
 */
template<class T, MemSpace M>
template<class InputIterator>
auto PieBuilder<T, M>::insert_back(InputIterator first, InputIterator last)
    -> PieSliceT
{
    CELER_EXPECT(std::distance(first, last) + this->storage().size()
                 <= this->max_pie_size());
    static_assert(M == MemSpace::host,
                  "Insertion currently works only for host memory");
    auto start = this->size();
    this->storage().insert(this->storage().end(), first, last);
    return {start, this->size()};
}

//---------------------------------------------------------------------------//
/*!
 * Insert the given list of elements at the end of the allocation.
 */
template<class T, MemSpace M>
auto PieBuilder<T, M>::insert_back(std::initializer_list<T> init) -> PieSliceT
{
    return this->insert_back(init.begin(), init.end());
}

//---------------------------------------------------------------------------//
/*!
 * Reserve space for the given number of elements.
 */
template<class T, MemSpace M>
void PieBuilder<T, M>::push_back(T el)
{
    CELER_EXPECT(this->storage().size() + 1 <= this->max_pie_size());
    static_assert(M == MemSpace::host,
                  "Insertion currently works only for host memory");
    this->storage().push_back(el);
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
template<class T, MemSpace M>
void PieBuilder<T, M>::resize(size_type size)
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
