//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Pie.hh
//---------------------------------------------------------------------------//
#pragma once

#include "PieTypes.hh"
#include "Types.hh"
#include "detail/PieImpl.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Reference a contiguous subset of items inside a Pie.
 *
 * \tparam T The value type of items to represent.
 *
 * The template parameter here isn't used directly -- it's more of a marker in
 * a class that contains it. The template parameter must match the
 * corresponding \c Pie type, and more importantly it's only assigned to one
 * particular pie. It doesn't have any persistent connection to its associated
 * pie and thus must be used carefully.
 *
 * The size type is plain "unsigned int" (32-bit in CUDA) rather than
 * \c celeritas::size_type (64-bit) because CUDA currently uses native 32-bit
 * pointer arithmetic. In general this should be the same type as the default
 * OpaqueId::value_type. It's possible that in large problems 4 billion
 * elements won't be enough (for e.g. cross sections), but in that case the
 * PieBuilder will throw an assertion during construction.
 *
 * \code
 * struct MyMaterial
 * {
 *     real_type number_density;
 *     PieSlice<ElementComponents> components;
 * };
 *
 * template<Ownership W, MemSpace M>
 * struct MyPies
 * {
 *     Pie<ElementComponents, W, M> components;
 *     Pie<MyMaterial, W, M> materials;
 * };
 * \endcode
 *
 * \todo Not sure if we ever have to directly iterate over the values, but if
 * we wanted to we could have this guy use \c detail::range_iter<unsigned int>
 * instead of unsigned int.
 */
template<class T>
class PieSlice
{
  public:
    //!@{
    using size_type = detail::pie_size_type;
    //!@}

  public:
    //! Default to an empty slice
    PieSlice() = default;

    // Construct with a particular range of element indices
    inline CELER_FUNCTION PieSlice(size_type start, size_type stop);

    //!@{
    //! Range of indices in the corresponding pie
    CELER_CONSTEXPR_FUNCTION size_type start() const { return start_; }
    CELER_CONSTEXPR_FUNCTION size_type stop() const { return stop_; }
    //!@}

    //! Whether the slice is empty
    CELER_CONSTEXPR_FUNCTION bool empty() const { return stop_ == start_; }

    //! Number of elements
    CELER_CONSTEXPR_FUNCTION size_type size() const { return stop_ - start_; }

  private:
    size_type start_{};
    size_type stop_{};
};

//---------------------------------------------------------------------------//
/*!
 * Manage generic array-like data ownership and transfer from host to device.
 *
 * Pies are constructed incrementally on the host, then copied (along with
 * their associated PieSlice ) to device. A Pie can act as a std::vector<T>,
 * DeviceVector<T>, Span<T>, or Span<const T>. The Spans can point to host or
 * device memory, but the MemSpace template argument protects against
 * accidental accesses from the wrong memory space.
 *
 * Each Pie object is usually accessed with a Slice, which references a
 * contiguous set of elements in the Pie. For example, setup code on the host
 * would extend the Pie with a series of vectors, the addition of which
 * returns a PieSlice that returns the equivalent data on host or device. This
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
 */
template<class T, Ownership W, MemSpace M>
class Pie
{
    using PieTraitsT = detail::PieTraits<T, W>;

  public:
    //!@{
    //! Type aliases
    using SpanT                = typename PieTraitsT::SpanT;
    using SpanConstT           = typename PieTraitsT::SpanConstT;
    using value_type           = typename PieTraitsT::value_type;
    using pointer              = typename PieTraitsT::pointer;
    using const_pointer        = typename PieTraitsT::const_pointer;
    using reference_type       = typename PieTraitsT::reference_type;
    using const_reference_type = typename PieTraitsT::const_reference_type;
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
    inline Pie(const Pie<T, W2, M2>& other);

    // Construct from another pie (mutable)
    template<Ownership W2, MemSpace M2>
    inline Pie(Pie<T, W2, M2>& other);

    //!@{
    //! Default assignment
    Pie& operator=(const Pie& other) = default;
    Pie& operator=(Pie&& other) = default;
    //!@}

    // Assign from another pie in the same memory space
    template<Ownership W2>
    inline Pie& operator=(const Pie<T, W2, M>& other);

    // Assign (mutable!) from another pie in the same memory space
    template<Ownership W2>
    inline Pie& operator=(Pie<T, W2, M>& other);

    //// ACCESS ////

    // Access a subset of the data with a slice
    inline CELER_FUNCTION SpanT operator[](const PieSlice<T>& ps) const;

    // Access a single element
    inline CELER_FUNCTION reference_type operator[](size_type i) const;

    // Direct accesors to underlying data
    CELER_CONSTEXPR_FUNCTION size_type     size() const;
    CELER_CONSTEXPR_FUNCTION bool          empty() const;
    CELER_CONSTEXPR_FUNCTION const_pointer data() const;
    CELER_CONSTEXPR_FUNCTION pointer       data();

  private:
    //// DATA ////

    detail::PieStorage<T, W, M> storage_{};

    //// FRIENDS ////

    template<class T2, Ownership W2, MemSpace M2>
    friend class Pie;

    template<class T2, MemSpace M2>
    friend class PieBuilder;

    //!@{
    // Private accessors for pie construction
    using StorageT = typename detail::PieStorage<T, W, M>::type;
    CELER_FORCEINLINE_FUNCTION const StorageT& storage() const
    {
        return storage_.data;
    }
    CELER_FORCEINLINE_FUNCTION StorageT& storage() { return storage_.data; }
    //@}
};

//---------------------------------------------------------------------------//
} // namespace celeritas

//---------------------------------------------------------------------------//
/*!
 * \def CELER_PIE_TYPE(CLSNAME, OWNERSHIP, MEMSPACE)
 *
 * Define type alias for a template class (usually a collection of pies) with
 * the given ownership and memory space.
 */
#define CELER_PIE_TYPE(CLSNAME, OWNERSHIP, MEMSPACE) \
    CLSNAME<::celeritas::Ownership::OWNERSHIP, ::celeritas::MemSpace::MEMSPACE>

//---------------------------------------------------------------------------//
/*!
 * \def CELER_PIE_STRUCT
 *
 * Define an anonymous struct that holds sets of values and references on
 * host and device. This is meant for collections of Pies in a struct
 * that's templated only on the ownership and memory space.
 *
 * Example:
 * \code
 * class FooParams
 * {
 *  public:
 *   using PieDeviceRef = CELER_PIE_TYPE(FooPies, const_reference, device);
 *
 *   const PieDeviceRef& device_pointers() const {
 *    return pies_.device_ref;
 *   }
 *  private:
 *   CELER_PIE_STRUCT(FooPies, const_reference) pies_;
 * };
 * \endcode
 */
#define CELER_PIE_STRUCT(CLSNAME, REFTYPE)                   \
    struct                                                   \
    {                                                        \
        CELER_PIE_TYPE(CLSNAME, value, host) host;           \
        CELER_PIE_TYPE(CLSNAME, value, device) device;       \
        CELER_PIE_TYPE(CLSNAME, REFTYPE, host) host_ref;     \
        CELER_PIE_TYPE(CLSNAME, REFTYPE, device) device_ref; \
    }

//---------------------------------------------------------------------------//

#include "Pie.i.hh"
