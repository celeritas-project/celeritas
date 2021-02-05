//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PieBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include <initializer_list>
#include <limits>
#include "Pie.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Helper class for constructing Pies.
 *
 * This is intended for use with host data but can also be used to resize
 * device pies. It's constructed with a reference to the host pie, and it
 * provides vector-like methods for extending it. The size *cannot* be
 * decreased because that would invalidate previously created \c PieSlice
 * items.
 *
 * \code
    auto pb = make_pie_builder(myintpie.host);
    pb.reserve(100);
    PieSlice<int> insertion = pb.extend(local_ints.begin(), local_ints.end());
    pb.push_back(123);
   \endcode

 * The PieBuilder can also be used to resize device-value pies without having
 * to allocate a host version and copy to device. (This is useful for state
 * allocations.)
 */
template<class T, MemSpace M>
class PieBuilder
{
  public:
    //!@{
    //! Type aliases
    using value_type = T;
    using PieT       = Pie<T, Ownership::value, M>;
    using PieSliceT  = PieSlice<T>;
    using PieSize    = typename PieSliceT::size_type;
    //!@}

  public:
    //! Construct from a pie
    explicit PieBuilder(PieT& pie) : pie_(pie) {}

    // Increase size to this capacity
    inline void resize(size_type count);

    // Reserve space
    inline void reserve(size_type count);

    // Extend with a series of elements, returning the range inserted
    template<class InputIterator>
    inline PieSliceT insert_back(InputIterator first, InputIterator last);

    // Extend with a series of elements from an initializer list
    inline PieSliceT insert_back(std::initializer_list<T> init);

    // Append a single element
    inline void push_back(value_type element);

    //! Number of elements in the pie
    PieSize size() const { return pie_.size(); }

  private:
    PieT& pie_;

    using StorageT = typename PieT::StorageT;
    StorageT&       storage() { return pie_.storage(); }
    const StorageT& storage() const { return pie_.storage(); }

    //! Maximum elements in a Pie, in native std::size_t
    static constexpr size_type max_pie_size()
    {
        return std::numeric_limits<PieSize>::max();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Helper class for constructing pie builders.
 *
 * (Will not be needed under C++20's new constructor lookups).
 */
template<class T, MemSpace M>
PieBuilder<T, M> make_pie_builder(Pie<T, Ownership::value, M>& pie)
{
    return PieBuilder<T, M>(pie);
}

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "PieBuilder.i.hh"
