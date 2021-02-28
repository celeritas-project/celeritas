//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Pie.hh
//---------------------------------------------------------------------------//
#pragma once

#include "OpaqueId.hh"
#include "PieTypes.hh"
#include "Range.hh"
#include "Types.hh"
#include "detail/PieImpl.hh"

namespace celeritas
{
//! Opaque ID representing a single element of a container.
template<class T>
using ItemId = OpaqueId<T, unsigned int>;

//---------------------------------------------------------------------------//
/*!
 * Reference a contiguous range of IDs corresponding to a slice of items.
 *
 * \tparam T The value type of items to represent.
 *
 * A ItemRange is a range of \c OpaqueId<T> that reference a range of values of
 * type \c T in a \c Pie . The ItemRange acts like a \c slice object in Python
 * when used on a Pie, returning a Span<T> of the underlying data.
 *
 * A ItemRange is only meaningful in connection with a particular Pie of type
 * T. It doesn't have any persistent connection to its associated pie and thus
 * must be used carefully.
 *
 * \todo It might also be good to have a `PieMap` -- mapping one OpaqueId to
 * another OpaqueId type (with just an offset value). This would be used for
 * example in physics, where \c ItemRange objects themselves are supposed to be
 * indexed into with a particular ID type.
 *
 * \code
 * struct MyMaterial
 * {
 *     real_type number_density;
 *     ItemRange<ElementComponents> components;
 * };
 *
 * template<Ownership W, MemSpace M>
 * struct MyData
 * {
 *     Pie<ElementComponents, W, M> components;
 *     Pie<MyMaterial, W, M> materials;
 * };
 * \endcode
 */
template<class T, class Size = unsigned int>
using ItemRange = Range<OpaqueId<T, Size>>;

//---------------------------------------------------------------------------//
/*!
 * Manage generic array-like data ownership and transfer from host to device.
 *
 * Data are constructed incrementally on the host, then copied (along with
 * their associated ItemRange ) to device. A Pie can act as a std::vector<T>,
 * DeviceVector<T>, Span<T>, or Span<const T>. The Spans can point to host or
 * device memory, but the MemSpace template argument protects against
 * accidental accesses from the wrong memory space.
 *
 * Each Pie object is usually accessed with a Slice, which references a
 * contiguous set of elements in the Pie. For example, setup code on the host
 * would extend the Pie with a series of vectors, the addition of which
 * returns a ItemRange that returns the equivalent data on host or device. This
 * methodology allows complex nested data structures to be built up quickly at
 * setup time without knowing the size requirements beforehand.
 *
 * Host-device functions and classes should use \c Pie with a reference or
 * const_reference Ownership, and the \c MemSpace::native type, which expects
 * device memory when compiled inside a CUDA file and host memory when used
 * inside a C++ source or test. (This design choice prevents a single CUDA file
 * from compiling separate host-compatible and device-compatible compute
 * kernels, but in the case of Celeritas this situation won't arise, because
 * we always want to build host code in C++ files for development ease and to
 * allow testing when CUDA is disabled.)
 *
 * \todo It would be easy to specialize the traits for the const_reference
 * ownership so that for device primitive data types (int, double) we access
 * via __ldg -- speeding up everywhere in the code without any invasive
 * changes. This is another good argument for using Pie instead of Span for
 * device-compatible helper classes (e.g. grid calculator).
 */
template<class T, Ownership W, MemSpace M, class I = ItemId<T>>
class Pie
{
    using PieTraitsT = detail::PieTraits<T, W>;

  public:
    //!@{
    //! Type aliases
    using SpanT                = typename PieTraitsT::SpanT;
    using SpanConstT           = typename PieTraitsT::SpanConstT;
    using pointer              = typename PieTraitsT::pointer;
    using const_pointer        = typename PieTraitsT::const_pointer;
    using reference_type       = typename PieTraitsT::reference_type;
    using const_reference_type = typename PieTraitsT::const_reference_type;
    using size_type            = typename I::size_type;
    using ItemIdT              = I;
    using ItemRangeT           = Range<ItemIdT>;
    //!@}

  public:
    //// CONSTRUCTION ////

    //!@{
    //! Default constructors
    Pie()           = default;
    Pie(const Pie&) = default;
    Pie(Pie&&)      = default;
    //!@}

    // Construct from another pie
    template<Ownership W2, MemSpace M2>
    inline Pie(const Pie<T, W2, M2, I>& other);

    // Construct from another pie (mutable)
    template<Ownership W2, MemSpace M2>
    inline Pie(Pie<T, W2, M2, I>& other);

    //!@{
    //! Default assignment
    Pie& operator=(const Pie& other) = default;
    Pie& operator=(Pie&& other) = default;
    //!@}

    // Assign from another pie in the same memory space
    template<Ownership W2>
    inline Pie& operator=(const Pie<T, W2, M, I>& other);

    // Assign (mutable!) from another pie in the same memory space
    template<Ownership W2>
    inline Pie& operator=(Pie<T, W2, M, I>& other);

    //// ACCESS ////

    // Access a subset of the data with a slice
    inline CELER_FUNCTION SpanT      operator[](ItemRangeT ps);
    inline CELER_FUNCTION SpanConstT operator[](ItemRangeT ps) const;

    // Access a single element
    inline CELER_FUNCTION reference_type       operator[](ItemIdT i);
    inline CELER_FUNCTION const_reference_type operator[](ItemIdT i) const;

    // Direct accesors to underlying data
    CELER_CONSTEXPR_FUNCTION size_type     size() const;
    CELER_CONSTEXPR_FUNCTION bool          empty() const;
    CELER_CONSTEXPR_FUNCTION const_pointer data() const;
    CELER_CONSTEXPR_FUNCTION pointer       data();

  private:
    //// DATA ////

    detail::PieStorage<T, W, M> storage_{};

    //// FRIENDS ////

    template<class T2, Ownership W2, MemSpace M2, class Id2>
    friend class Pie;

    template<class T2, MemSpace M2, class Id2>
    friend class PieBuilder;

  protected:
    //!@{
    // Private accessors for pie construction/access
    using StorageT = typename detail::PieStorage<T, W, M>::type;
    CELER_FORCEINLINE_FUNCTION const StorageT& storage() const
    {
        return storage_.data;
    }
    CELER_FORCEINLINE_FUNCTION StorageT& storage() { return storage_.data; }
    //@}
};

//! Pie for data of type T but indexed by ThreadId for use in States
template<class T, Ownership W, MemSpace M>
using StatePie = Pie<T, W, M, ThreadId>;

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "Pie.i.hh"
